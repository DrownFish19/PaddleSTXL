import os
import pickle

import numpy as np
import paddle.io as io

from dataset.data_utils import ScalerMinMax


class TrafficFlowDataset(io.Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """

    def __init__(self, training_args, data_type="train"):
        super().__init__()
        self.training_args = training_args

        # step 1: load data
        # [T, N, D]
        if data_type == "train":
            self.data_path = training_args.train_data_path
        elif data_type == "val":
            self.data_path = training_args.val_data_path
        elif data_type == "test":
            self.data_path = training_args.test_data_path

        origin_data = np.load(self.data_path)["data"].astype(np.float32)
        self.data_mask = origin_data[:, :, :1]
        self.data_input = origin_data[:, :, 1:2]
        self.seq_len, self.num_nodes, self.dims = self.data_input.shape
        self.data_type = data_type

        # step 3: scale data
        if training_args.scale:
            # scale the input data
            if os.path.exists(self.training_args.scaler_data_path):
                self.scaler = pickle.load(self.training_args.scaler_data_path)
            else:
                self.scaler = ScalerMinMax()
                if data_type == "train":
                    train_data = origin_data
                else:
                    train_data = np.load(self.training_args.train_data_path)["data"]
                    train_data = train_data.astype(np.float32)
                self.scaler.fit(train_data[:, :, 1:2].reshape(-1, self.dims))
                with open(self.training_args.scaler_data_path, "wb") as f:
                    pickle.dump(self.scaler, f)

            self.data_input = self.scaler.transform(self.data_input).reshape(
                self.seq_len, self.num_nodes, self.dims
            )

    def __getitem__(self, index):
        his_begin = index
        his_end = his_begin + self.training_args.his_len
        tgt_begin = his_end
        tgt_end = tgt_begin + self.training_args.tgt_len
        # [T, N, F]
        his = self.data_input[his_begin:his_end]
        his_mask = self.data_mask[his_begin:his_end]
        tgt = self.data_input[tgt_begin:tgt_end]
        tgt_mask = self.data_mask[tgt_begin:tgt_end]
        return his, his_mask, tgt, tgt_mask

    def __len__(self):
        return (
            self.data_input.shape[0]
            - self.training_args.his_len
            - self.training_args.tgt_len
        )

    def inverse_transform(self, data, axis=None):
        if self.training_args.scale:
            return self.scaler.inverse_transform(data, axis)
        else:
            return data


if __name__ == "__main__":
    from args import args

    dataset_train = TrafficFlowDataset(args, "train")
    dataset_val = TrafficFlowDataset(args, "val")
    dataset_test = TrafficFlowDataset(args, "test")
    dataloader = io.DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=4)
    for i, data in enumerate(dataloader):
        print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
