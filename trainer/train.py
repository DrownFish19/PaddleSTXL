import contextlib
import os
from time import time

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
import paddle.io as io
import paddle.nn as nn
import paddle.optimizer as optimizer
import sklearn.metrics as skmetrics
from hssinfo import cluster

from args import args
from dataset import TrafficFlowDataset
from models import STNXL, GraphST
from utils import Logger, masked_mape_np


@contextlib.contextmanager
def amp_guard_context(fp16=False):
    if fp16:
        return paddle.amp.auto_cast(level="O2")
    else:
        return contextlib.nullcontext()


class Trainer:
    def __init__(self, training_args):
        self.training_args = training_args

        self.folder_dir = (
            f"MAE_{training_args.model_name}_elayer{training_args.encoder_num_layers}_"
            + f"dlayer{training_args.decoder_num_layers}_head{training_args.head}_dm{training_args.d_model}_"
            + f"einput{training_args.encoder_input_size}_dinput{training_args.decoder_input_size}_"
            + f"doutput{training_args.decoder_output_size}_drop{training_args.dropout}_"
            + f"lr{training_args.learning_rate}_wd{training_args.weight_decay}_bs{training_args.batch_size}_"
            + f"topk{training_args.top_k}_att{training_args.attention}_trepoch{training_args.train_epochs}_"
            + f"finepoch{training_args.finetune_epochs}"
        )

        self.save_path = os.path.join(
            "experiments", training_args.dataset_name, self.folder_dir
        )
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = Logger("CorrSTN", os.path.join(self.save_path, "log.txt"))

        if training_args.start_epoch == 0:
            self.logger.info(f"create params directory {self.save_path}")
        elif training_args.start_epoch > 0:
            self.logger.info(f"train from params directory {self.save_path}")

        self.logger.info(f"save folder: {self.folder_dir}")
        self.logger.info(f"save path  : {self.save_path}")
        self.logger.info(f"log  file  : {self.logger.log_file}")

        self.finetune = False

        self._build_data()
        self._build_model()
        self._build_optimizer()
        if training_args.distribute:
            self._build_distribute()

    def _build_data(self):
        self.train_dataset = TrafficFlowDataset(self.training_args, "train")
        self.val_dataset = TrafficFlowDataset(self.training_args, "val")
        self.test_dataset = TrafficFlowDataset(self.training_args, "test")

        self.train_dataloader = io.DataLoader(
            self.train_dataset, batch_size=self.training_args.batch_size, shuffle=True
        )
        self.eval_dataloader = io.DataLoader(
            self.val_dataset, batch_size=self.training_args.batch_size * 32
        )
        self.test_dataloader = io.DataLoader(
            self.test_dataset, batch_size=self.training_args.batch_size * 32
        )

        self.encoder_idx = []
        if self.training_args.his_len >= 2016:
            self.fix_week = paddle.arange(
                start=self.training_args.his_len - 2016,
                end=self.training_args.his_len - 2016 + 12,
            )
            self.encoder_idx.append(self.fix_week)
        if self.training_args.his_len >= 288:
            self.fix_day = paddle.arange(
                start=self.training_args.his_len - 288,
                end=self.training_args.his_len - 288 + 12,
            )
            self.encoder_idx.append(self.fix_day)
        if self.training_args.his_len >= 12:
            self.fix_hour = paddle.arange(
                start=self.training_args.his_len - 12,
                end=self.training_args.his_len,
            )
            self.encoder_idx.append(self.fix_hour)
        self.encoder_idx = paddle.concat(self.encoder_idx)

    def _build_model(self):
        self.graph = GraphST(args=self.training_args, build=False)
        self.cluster = cluster(
            paddle.arange(self.graph.node_nums),
            paddle.to_tensor(self.graph.edge_dst_idx, dtype=paddle.int32),
            paddle.to_tensor(self.graph.edge_src_idx, dtype=paddle.int32),
            paddle.to_tensor(self.graph.edge_weights, dtype=paddle.float32),
        )

        nn.initializer.set_global_initializer(
            nn.initializer.XavierUniform(), nn.initializer.XavierUniform()
        )

        self.net = STNXL(self.training_args, graph=self.graph)
        if self.training_args.continue_training:
            params_filename = os.path.join(self.save_path, "epoch_best.params")
            self.net.set_state_dict(paddle.load(params_filename))
            self.logger.info(f"load weight from: {params_filename}")

        if self.training_args.fp16:
            self.net = paddle.amp.decorate(models=self.net, level="O2")
            self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        self.logger.info(self.net)

        total_param = 0
        self.logger.info("Net's state_dict:")
        for param_tensor in self.net.state_dict():
            self.logger.info(
                f"{param_tensor} \t {self.net.state_dict()[param_tensor].shape}"
            )
            total_param += np.prod(self.net.state_dict()[param_tensor].shape)
        self.logger.info(f"Net's total params: {total_param}.")

        self.criterion1 = nn.L1Loss()  # 定义损失函数
        self.criterion2 = nn.MSELoss()  # 定义损失函数

    def _build_optimizer(self):
        # 定义优化器，传入所有网络参数
        self.optimizer = optimizer.AdamW(
            parameters=self.net.parameters(),
            learning_rate=self.training_args.lr_scheduler,
            weight_decay=self.training_args.weight_decay,
            multi_precision=True,
        )

        self.logger.info("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            self.logger.info(f"{var_name} \t {self.optimizer.state_dict()[var_name]}")

    def _build_distribute(self):
        # 二、初始化 Fleet 环境
        fleet.init(is_collective=True)

        # 三、构建分布式训练使用的网络模型
        self.net = fleet.distributed_model(self.net)

        # 四、构建分布式训练使用的优化器
        self.optimizer = fleet.distributed_optimizer(self.optimizer)

        # 五、构建分布式训练使用的数据集
        train_sampler = io.DistributedBatchSampler(
            self.train_dataset, batch_size=self.training_args.batch_size, shuffle=True
        )
        eval_sampler = io.DistributedBatchSampler(
            self.val_dataset, batch_size=self.training_args.batch_size
        )

        test_sampler = io.DistributedBatchSampler(
            self.test_dataset, batch_size=self.training_args.batch_size
        )

        self.train_dataloader = io.DataLoader(
            self.train_dataset, batch_sampler=train_sampler, num_workers=2
        )
        self.eval_dataloader = io.DataLoader(
            self.val_dataset, batch_sampler=eval_sampler, num_workers=2
        )
        self.test_dataloader = io.DataLoader(
            self.test_dataset, batch_sampler=test_sampler, num_workers=2
        )

    def train(self):
        self.logger.info("start train...")

        s_time = time()
        best_eval_loss = np.inf
        best_epoch = 0
        global_step = 0
        epoch = self.training_args.start_epoch

        while (
            epoch < self.training_args.train_epochs + self.training_args.finetune_epochs
        ):
            # finetune => load best training model
            if epoch == self.training_args.train_epochs:
                self._init_finetune()
                self.compute_test_loss()

            self.net.train()  # ensure dropout layers are in train mode
            tr_s_time = time()
            epoch_step = 0
            self.lr_scheduler.step()
            for batch_index, batch_data in enumerate(self.train_dataloader):
                src, tgt = batch_data
                _, training_loss = self.train_one_step(src, tgt)
                # self.logger.info(f"training_loss: {training_loss.numpy()}")
                epoch_step += 1
                global_step += 1
            self.logger.info(f"learning_rate: {self.optimizer.get_lr()}")
            self.logger.info(f"epoch: {epoch}, train time cost:{time() - tr_s_time}")
            self.logger.info(f"epoch: {epoch}, total time cost:{time() - s_time}")

            # apply model on the validation data set
            eval_loss = self.compute_eval_loss()
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                self.logger.info(f"best_epoch: {best_epoch}")
                self.logger.info(f"eval_loss: {eval_loss}")
                params_filename = os.path.join(self.save_path, "epoch_best.params")
                paddle.save(self.net.state_dict(), params_filename)
                self.logger.info(f"save parameters to file: {params_filename}")
            epoch += 1

        self.logger.info(f"best epoch: {best_epoch}")
        self.logger.info("apply the best val model on the test dataset ...")

        # params_filename = os.path.join(self.save_path, f"epoch_{best_epoch}.params")
        params_filename = os.path.join(self.save_path, "epoch_best.params")
        self.logger.info(f"load weight from: {params_filename}")
        self.net.set_state_dict(paddle.load(params_filename))
        self.compute_test_loss()

    def _init_finetune(self):
        self.logger.info("Start FineTune Training")
        params_filename = os.path.join(self.save_path, "epoch_best.params")
        self.net.set_state_dict(paddle.load(params_filename))
        self.logger.info(f"load weight from: {params_filename}")
        self.optimizer._learning_rate = self.training_args.learning_rate * 0.1
        self.finetune = True

    def train_one_step(self, src, tgt):
        """_summary_

        Args:
            src (_type_): [B,N,T,D]
            tgt (_type_): [B,N,T,D]

        Returns:
            _type_: _description_
        """
        self.net.train()
        encoder_input = paddle.index_select(src, self.encoder_idx, axis=2)

        with amp_guard_context(self.training_args.fp16):
            if not self.finetune:
                decoder_input = paddle.concat(
                    [src[:, :, -1:, :], tgt[:, :, :-1, :]], axis=-2
                )
                decoder_output = self.net(
                    src=encoder_input, src_idx=self.encoder_idx, tgt=decoder_input
                )
            else:
                decoder_start_inputs = encoder_input[:, :, -1:, :]
                decoder_input_list = [decoder_start_inputs]
                encoder_output = self.net.encode(encoder_input, self.encoder_idx)

                for step in range(self.training_args.tgt_len):
                    decoder_inputs = paddle.concat(decoder_input_list, axis=2)
                    decoder_output = self.net.decode(decoder_inputs, encoder_output)
                    decoder_input_list = [decoder_start_inputs, decoder_output]

            # decoder_output = paddle.where(tgt == -1, tgt, decoder_output)
            loss = self.criterion1(decoder_output, tgt)
        if self.net.training:
            if self.training_args.fp16:
                scaled = self.scaler.scale(loss)  # loss 缩放，乘以系数 loss_scaling
                scaled.backward()  # 反向传播
                self.scaler.step(self.optimizer)  # 更新参数（参数梯度先除系数 loss_scaling 再更新参数）
                self.scaler.update()  # 基于动态 loss_scaling 策略更新 loss_scaling 系数
                self.optimizer.clear_grad(set_to_zero=False)
            else:
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
        return decoder_output, loss

    def eval_one_step(self, src, tgt):
        self.net.eval()
        encoder_input = paddle.index_select(src, self.encoder_idx, axis=2)
        with amp_guard_context(self.training_args.fp16):
            decoder_start_inputs = encoder_input[:, :, -1:, :]
            decoder_input_list = [decoder_start_inputs]

            encoder_output = self.net.encode(encoder_input, self.encoder_idx)

            for step in range(self.training_args.tgt_len):
                decoder_inputs = paddle.concat(decoder_input_list, axis=2)
                decoder_output = self.net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, decoder_output]

            loss = self.criterion1(decoder_output, tgt)

        return decoder_output, loss

    def test_one_step(self, src, tgt):
        self.net.eval()
        encoder_input = paddle.index_select(src, self.encoder_idx, axis=2)
        with amp_guard_context(self.training_args.fp16):
            decoder_start_inputs = encoder_input[:, :, -1:, :]
            decoder_input_list = [decoder_start_inputs]

            encoder_output = self.net.encode(encoder_input, self.encoder_idx)

            for step in range(self.training_args.tgt_len):
                decoder_inputs = paddle.concat(decoder_input_list, axis=2)
                decoder_output = self.net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, decoder_output]

            loss = self.criterion1(decoder_output, tgt)

        return decoder_output, loss

    def compute_eval_loss(self):
        with paddle.no_grad():
            all_eval_loss = []  # 记录了所有batch的loss
            start_time = time()
            for batch_index, batch_data in enumerate(self.eval_dataloader):
                src, tgt = batch_data
                predict_output, eval_loss = self.eval_one_step(src, tgt)
                all_eval_loss.append(eval_loss.numpy())

            eval_loss = np.mean(all_eval_loss)
            self.logger.info(f"eval cost time: {time() - start_time}s")
            self.logger.info(f"eval_loss: {eval_loss}")
        return eval_loss

    def compute_test_loss(self):
        with paddle.no_grad():
            preds = []
            tgts = []
            start_time = time()
            for batch_index, batch_data in enumerate(self.test_dataloader):
                src, tgt = batch_data
                predict_output, _ = self.test_one_step(src, tgt)

                preds.append(predict_output.detach().numpy())
                tgts.append(tgt.detach().numpy())
            self.logger.info(f"test time on whole data: {time() - start_time}s")

            preds = np.concatenate(preds, axis=0)  # [B,N,T,1]
            trues = np.concatenate(tgts, axis=0)  # [B,N,T,F]
            preds = self.test_dataset.inverse_transform(preds, axis=-1)  # [B,N,T,1]
            trues = self.test_dataset.inverse_transform(trues, axis=-1)  # [B,N,T,1]

            self.logger.info(f"preds: {str(preds.shape)}")
            self.logger.info(f"tgts: {trues.shape}")

            # 计算误差
            excel_list = []
            prediction_length = trues.shape[2]

            for i in range(prediction_length):
                assert preds.shape[0] == trues.shape[0]
                mae = skmetrics.mean_absolute_error(
                    trues[:, :, i, 0], preds[:, :, i, 0]
                )
                rmse = (
                    skmetrics.mean_squared_error(trues[:, :, i, 0], preds[:, :, i, 0])
                    ** 0.5
                )
                mape = masked_mape_np(trues[:, :, i, 0], preds[:, :, i, 0], 0)
                self.logger.info(f"{i} MAE: {mae}")
                self.logger.info(f"{i} RMSE: {rmse}")
                self.logger.info(f"{i} MAPE: {mape}")
                excel_list.extend([mae, rmse, mape])

            # print overall results
            mae = skmetrics.mean_absolute_error(
                trues.reshape(-1, 1), preds.reshape(-1, 1)
            )
            rmse = (
                skmetrics.mean_squared_error(trues.reshape(-1, 1), preds.reshape(-1, 1))
                ** 0.5
            )
            mape = masked_mape_np(trues.reshape(-1, 1), preds.reshape(-1, 1), 0)
            self.logger.info(f"all MAE: {mae}")
            self.logger.info(f"all RMSE: {rmse}")
            self.logger.info(f"all MAPE: {mape}")
            excel_list.extend([mae, rmse, mape])
            self.logger.info(excel_list)

    def run_test(self):
        params_filename = os.path.join(self.save_path, "epoch_best.params")
        self.logger.info(f"load weight from: {params_filename}")
        self.net.set_state_dict(paddle.load(params_filename))
        self.compute_test_loss()


if __name__ == "__main__":
    trainer = Trainer(training_args=args)
    trainer.train()
    trainer.run_test()
