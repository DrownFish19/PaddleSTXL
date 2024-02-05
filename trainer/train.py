import contextlib
import os
from time import time

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.io as io
import paddle.nn as nn
import paddle.optimizer as optimizer
import sklearn.metrics as skmetrics
import tqdm
import visualdl

from dataset import SpatialGraph, TrafficFlowDataset
from models import ASTGCN, STGCN, STLSTM, STNXL
from utils import CosineAnnealingWithWarmupDecay, Logger, masked_mape_np

MODEL_PARAMS = "epoch_best.params"
OPT_PARAMS = "epoch_best.pdopt"
GRAPH_PARAMS = "epoch_best.graph"


def amp_guard_context(fp16=False):
    if fp16:
        return paddle.amp.auto_cast(enable=True, level="O2")
    else:
        return contextlib.nullcontext()


class Trainer:
    def __init__(self, training_args):
        dist.init_parallel_env()

        self.training_args = training_args
        self.folder_dir = (
            f"MAE_{training_args.model_name}_elayer{training_args.encoder_num_layers}_"
            + f"dlayer{training_args.decoder_num_layers}_head{training_args.head}_dm{training_args.d_model}_"
            + f"einput{training_args.input_size}_dinput{training_args.input_size}_"
            + f"doutput{training_args.decoder_output_size}_drop{training_args.dropout}_"
            + f"lr{training_args.learning_rate}_wd{training_args.weight_decay}_bs{training_args.batch_size}_"
            + f"topk{training_args.node_top_k}_att{training_args.attention}_trepoch{training_args.train_epochs}_"
            + f"finepoch{training_args.finetune_epochs}"
        )

        self.save_path = os.path.join(
            "experiments", training_args.dataset_name, self.folder_dir
        )
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = Logger(
            f"{self.training_args.model_name}", os.path.join(self.save_path, "log.txt")
        )
        self.writer = visualdl.LogWriter(
            logdir=os.path.join(self.save_path, "visualdl")
        )

        self.logger.info(f"save folder: {self.folder_dir}")
        self.logger.info(f"save path  : {self.save_path}")
        self.logger.info(f"log  file  : {self.logger.log_file}")

        args_message = "\n".join(
            [f"{k:<20}: {v}" for k, v in vars(training_args).items()]
        )
        self.logger.info(f"training_args  : \n{args_message}")

        self._build_data()
        self._build_model()
        self._build_optimizer()

        if training_args.start_epoch == 0:
            self.logger.info(f"create params directory {self.save_path}")
        elif training_args.start_epoch > 0:
            self._load_params(epoch=training_args.start_epoch)
            self.logger.info(f"train from params directory {self.save_path}")

        if self.training_args.continue_training:
            self._load_best_params()
            self.logger.info(f"train from best params directory {self.save_path}")

    def _build_data(self):
        self.train_dataset = TrafficFlowDataset(self.training_args, "train")
        self.val_dataset = TrafficFlowDataset(self.training_args, "val")
        self.test_dataset = TrafficFlowDataset(self.training_args, "test")

        train_sampler = io.DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        eval_sampler = io.DistributedBatchSampler(
            self.val_dataset, batch_size=self.training_args.batch_size, drop_last=True
        )
        test_sampler = io.DistributedBatchSampler(
            self.test_dataset, batch_size=self.training_args.batch_size, drop_last=True
        )
        self.train_dataloader = io.DataLoader(
            self.train_dataset, batch_sampler=train_sampler
        )
        self.eval_dataloader = io.DataLoader(
            self.val_dataset, batch_sampler=eval_sampler
        )
        self.test_dataloader = io.DataLoader(
            self.test_dataset, batch_sampler=test_sampler
        )

    def _build_model(self):
        if os.path.exists(self.training_args.adj_path):
            self.graph = SpatialGraph(args=self.training_args, build=False)
        else:
            self.graph = SpatialGraph(args=self.training_args, build=True)
        if "PaddleSTXL" in self.training_args.model_name:
            self.graph.build_group_graph(n=2)

        nn.initializer.set_global_initializer(
            nn.initializer.XavierUniform(),
            nn.initializer.ConstantInitializer(0.0),
        )

        if "PaddleSTXL" in self.training_args.model_name:
            self.net = STNXL(self.training_args, graph=self.graph)
        elif "PaddleSTLSTM" in self.training_args.model_name:
            self.net = STLSTM(self.training_args)
        elif "PaddleSTGCN" in self.training_args.model_name:
            self.net = STGCN(self.training_args, graph=self.graph)
        elif "PaddleASTGCN" in self.training_args.model_name:
            self.net = ASTGCN(self.training_args, graph=self.graph)

        if self.training_args.fp16:
            self.net = paddle.amp.decorate(models=self.net, level="O2")
            self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        if self.training_args.distribute:
            self.net = paddle.DataParallel(self.net)
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
        self.lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.training_args.learning_rate,
            min_lr=self.training_args.learning_rate * 0.1,
            warmup_step=self.training_args.warmup_step,
            decay_step=self.training_args.decay_step,
        )

        self.optimizer = optimizer.AdamW(
            parameters=self.net.parameters(),
            learning_rate=self.lr_scheduler,
            weight_decay=self.training_args.weight_decay,
            multi_precision=True,
        )

        self.logger.info("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            self.logger.info(f"{var_name} \t {self.optimizer.state_dict()[var_name]}")

    def _save_best_params(self):
        params_name = os.path.join(self.save_path, MODEL_PARAMS)
        params_opt_name = os.path.join(self.save_path, OPT_PARAMS)
        params_graph_name = os.path.join(self.save_path, GRAPH_PARAMS)
        paddle.save(self.net.state_dict(), params_name)
        paddle.save(self.optimizer.state_dict(), params_opt_name)
        self.graph.save_graph(params_graph_name)
        self.logger.info(f"save parameters to file: {params_name}")

    def _save_params(self, epoch):
        model_name = f"epoch_{epoch}.params"
        opt_name = f"epoch_{epoch}.pdopt"
        graph_name = f"epoch_{epoch}.graph"
        group_name = f"epoch_{epoch}.group"
        mapping_name = f"epoch_{epoch}.mapping"
        decoder_tensor_name = f"epoch_{epoch}.decoder"
        params_name = os.path.join(self.save_path, model_name)
        params_opt_name = os.path.join(self.save_path, opt_name)
        params_graph_name = os.path.join(self.save_path, graph_name)
        params_group_name = os.path.join(self.save_path, group_name)
        params_mapping_name = os.path.join(self.save_path, mapping_name)
        params_decoder_name = os.path.join(self.save_path, decoder_tensor_name)
        paddle.save(self.net.state_dict(), params_name)
        paddle.save(self.optimizer.state_dict(), params_opt_name)
        self.graph.save_graph(params_graph_name)
        self.graph.save_group_graph(params_group_name)
        self.graph.save_group_mapping(params_mapping_name)
        if isinstance(self.net, paddle.DataParallel):
            np.save(params_decoder_name, self.net._layers.decoder_output.numpy())
        else:
            np.save(params_decoder_name, self.net.decoder_output.numpy())
        self.logger.info(f"save parameters to file: {params_name}")

    def _load_best_params(self):
        params_name = os.path.join(self.save_path, MODEL_PARAMS)
        params_opt_name = os.path.join(self.save_path, OPT_PARAMS)
        params_graph_name = os.path.join(self.save_path, GRAPH_PARAMS)
        self.optimizer.set_state_dict(paddle.load(params_opt_name))
        if isinstance(self.net, paddle.DataParallel):
            self.net._layers.set_state_dict(paddle.load(params_name))
            self.net._layers.load_graph(params_graph_name)
        else:
            self.net.set_state_dict(paddle.load(params_name))
            self.net.load_graph(params_graph_name)
        self.logger.info(f"load weight from: {params_name}")

    def _load_params(self, epoch):
        model_name = f"epoch_{epoch}.params"
        opt_name = f"epoch_{epoch}.pdopt"
        graph_name = f"epoch_{epoch}.graph"
        params_name = os.path.join(self.save_path, model_name)
        params_opt_name = os.path.join(self.save_path, opt_name)
        params_graph_name = os.path.join(self.save_path, graph_name)
        self.optimizer.set_state_dict(paddle.load(params_opt_name))
        if isinstance(self.net, paddle.DataParallel):
            self.net._layers.set_state_dict(paddle.load(params_name))
            self.net._layers.load_graph(params_graph_name)
        else:
            self.net.set_state_dict(paddle.load(params_name))
            self.net.load_graph(params_graph_name)
        self.logger.info(f"load weight from: {params_name}")

    def train(self):
        self.logger.info("start train...")
        s_time = time()
        best_eval_loss = np.inf
        best_epoch, global_step = 0, 0

        start_update_graph = self.training_args.start_epoch
        stop_update_graph = (
            self.training_args.train_epochs - self.training_args.finetune_epochs
        )
        for epoch in range(
            self.training_args.start_epoch, self.training_args.train_epochs
        ):
            self.net.train()  # ensure dropout layers are in train mode
            tr_s_time = time()
            epoch_step = 0
            self.lr_scheduler.step()
            self.train_dataloader = tqdm.tqdm(
                self.train_dataloader, disable=not self.training_args.tqdm
            )
            for batch_data in self.train_dataloader:
                _, training_loss = self.train_one_step(*batch_data)
                self.writer.add_scalar("train/loss", training_loss, global_step)
                epoch_step += 1
                global_step += 1

            self.logger.info(f"learning_rate: {self.optimizer.get_lr()}")
            self.logger.info(f"epoch: {epoch}, train time cost:{time() - tr_s_time}")
            self.logger.info(f"epoch: {epoch}, total time cost:{time() - s_time}")

            if epoch % self.training_args.eval_interval_epochs == 0 and epoch > 0:
                eval_loss = self.eval()
                self._save_params(epoch)

                if dist.get_rank() == 0:
                    self.writer.add_scalar("eval/loss", eval_loss, epoch)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_epoch = epoch
                        self.logger.info(f"best_epoch: {best_epoch}")
                        self.logger.info(f"eval_loss: {eval_loss}")
                        self._save_best_params()
                        # self.test()

            if epoch == stop_update_graph:
                self.logger.info(
                    "stop update graph and group and start to finetune the model ..."
                )
                self._load_best_params()

            if (
                epoch % self.training_args.update_graph_epochs == 0
                and epoch > start_update_graph
                and epoch < stop_update_graph
            ):
                with amp_guard_context(self.training_args.fp16):
                    if isinstance(self.net, paddle.DataParallel):
                        self.net._layers.update_graph()
                    else:
                        self.net.update_graph()

        self.logger.info(f"best epoch: {best_epoch}")
        self.logger.info("apply the best val model on the test dataset ...")
        self._load_best_params()
        self.test()

    def eval(self):
        self.logger.info("start to compute eval loss ...")
        with paddle.no_grad():
            all_eval_loss = []  # 记录了所有batch的loss
            start_time = time()
            self.eval_dataloader = tqdm.tqdm(
                self.eval_dataloader, disable=not self.training_args.tqdm
            )
            for batch_data in self.eval_dataloader:
                predict_output, eval_loss = self.eval_one_step(*batch_data)
                all_eval_loss.append(eval_loss.numpy())

            eval_loss = np.mean(all_eval_loss)
            all_eval_loss = []
            if dist.get_world_size() > 1:
                dist.all_gather_object(all_eval_loss, eval_loss)
                paddle.device.cuda.empty_cache()
                eval_loss = np.mean(
                    [all_eval_loss[i] for i in range(dist.get_world_size())]
                )
                self.logger.info(f"eval cost time: {time() - start_time}s")
                self.logger.info(f"eval_loss: {eval_loss}")
        return eval_loss

    def test(self):
        self.logger.info("start to compute test loss ...")
        with paddle.no_grad():
            preds, tgts = [], []
            start_time = time()
            self.test_dataloader = tqdm.tqdm(
                self.test_dataloader, disable=not self.training_args.tqdm
            )
            for batch_data in self.test_dataloader:
                his, his_mask, his_idx, tgt, tgt_mask, tgt_idx = batch_data
                predict_output, _ = self.test_one_step(*batch_data)

                preds.append(predict_output.detach().numpy())
                tgts.append(tgt.detach().numpy())
            self.logger.info(f"test time on whole data: {time() - start_time}s")

            preds = np.concatenate(preds, axis=0)  # [B,N,T,1]
            trues = np.concatenate(tgts, axis=0)  # [B,N,T,F]
            preds = self.test_dataset.inverse_transform(preds, axis=-1)  # [B,N,T,1]
            trues = self.test_dataset.inverse_transform(trues, axis=-1)  # [B,N,T,1]

            self.logger.info(f"preds: {str(preds.shape)}")
            self.logger.info(f"tgts: {trues.shape}")

            if dist.get_world_size() > 1:
                all_preds = []
                all_trues = []
                dist.all_gather_object(all_preds, preds)
                dist.all_gather_object(all_trues, trues)
                paddle.device.cuda.empty_cache()
                if dist.get_rank() == 0:
                    preds = np.concatenate(
                        [all_preds[i] for i in range(dist.get_world_size())], axis=0
                    )
                    trues = np.concatenate(
                        [all_trues[i] for i in range(dist.get_world_size())], axis=0
                    )
                else:
                    return

            # 计算误差
            excel_list = []
            prediction_length = trues.shape[1]

            for i in range(prediction_length):
                assert preds.shape[0] == trues.shape[0]
                pred = preds[:, i, :, 0]
                tgt = trues[:, i, :, 0]
                mae = skmetrics.mean_absolute_error(tgt, pred)
                rmse = skmetrics.mean_squared_error(tgt, pred) ** 0.5
                mape = masked_mape_np(tgt, pred, 0)
                self.logger.info(f"{i} MAE: {mae}")
                self.logger.info(f"{i} RMSE: {rmse}")
                self.logger.info(f"{i} MAPE: {mape}")
                excel_list.extend([mae, rmse, mape])

            # print overall results
            trues = trues.reshape(-1, 1)
            preds = preds.reshape(-1, 1)
            mae = skmetrics.mean_absolute_error(trues, preds)
            rmse = skmetrics.mean_squared_error(trues, preds) ** 0.5
            mape = masked_mape_np(trues, preds, 0)
            self.logger.info(f"all MAE: {mae}")
            self.logger.info(f"all RMSE: {rmse}")
            self.logger.info(f"all MAPE: {mape}")
            excel_list.extend([mae, rmse, mape])
            self.logger.info(excel_list)

    def train_one_step(self, his, his_mask, his_idx, tgt, tgt_mask, tgt_idx):
        """_summary_

        Args:
            src (_type_): [B,N,T,D]
            tgt (_type_): [B,N,T,D]

        Returns:
            _type_: _description_
        """
        self.net.train()
        with amp_guard_context(self.training_args.fp16):
            decoder_input = paddle.zeros_like(tgt)
            decoder_output = self.net(
                src=his, src_idx=his_idx, tgt=decoder_input, tgt_idx=tgt_idx
            )
            decoder_output = decoder_output * tgt_mask
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

    def eval_one_step(self, his, his_mask, his_idx, tgt, tgt_mask, tgt_idx):
        self.net.eval()
        with amp_guard_context(self.training_args.fp16):
            decoder_input = paddle.zeros_like(tgt)
            decoder_output = self.net(
                src=his, src_idx=his_idx, tgt=decoder_input, tgt_idx=tgt_idx
            )
            decoder_output = decoder_output * tgt_mask
            loss = self.criterion1(decoder_output, tgt)
        return decoder_output, loss

    def test_one_step(self, his, his_mask, his_idx, tgt, tgt_mask, tgt_idx):
        self.net.eval()
        with amp_guard_context(self.training_args.fp16):
            decoder_input = paddle.zeros_like(tgt)
            decoder_output = self.net(
                src=his, src_idx=his_idx, tgt=decoder_input, tgt_idx=tgt_idx
            )
            decoder_output = decoder_output * tgt_mask
            loss = self.criterion1(decoder_output, tgt)
        return decoder_output, loss

    def run_test(self):
        self._load_best_params()
        self.test()
