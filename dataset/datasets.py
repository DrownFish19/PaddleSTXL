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

    def __init__(self, training_args, data_type="train", scaler=None):
        super().__init__()
        self.training_args = training_args
        # step 1: load data
        # [T, N, D]
        origin_data = np.load(training_args.data_path)["data"].astype(np.float32)
        self.data_mask = origin_data[:, :, :1]
        self.data_input = origin_data[:, :, 1:2]
        self.seq_len, self.num_nodes, self.dims = self.data_input.shape

        # step 2: compute data ratio
        self.train_ratio, self.val_ratio, self.test_ratio = map(
            int, training_args.split.split(":")
        )
        sum_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        self.train_ratio /= sum_ratio
        self.val_ratio /= sum_ratio
        self.test_ratio /= sum_ratio

        self.train_size = int(self.seq_len * self.train_ratio)
        self.val_size = int(self.seq_len * self.val_ratio)
        self.test_size = int(self.seq_len * self.test_ratio)
        self.data_type = data_type

        # step 3: scale data
        if training_args.scale:
            # scale the input data
            if scaler is None:
                self.scaler = ScalerMinMax()
                train_data = self.data_input[: self.train_size, :, :]
                self.scaler.fit(train_data.reshape(-1, self.dims))
            else:
                self.scaler = scaler
            self.data_input = self.scaler.transform(self.data_input).reshape(
                self.seq_len, self.num_nodes, self.dims
            )

        # step 4: split data
        train_start, train_end = 0, self.train_size
        val_start, val_end = train_end, train_end + self.val_size
        test_start, test_end = val_end, min(val_end + self.test_size, self.seq_len)
        if self.data_type == "train":
            self.data_input = self.data_input[train_start:train_end]
            self.data_mask = self.data_mask[train_start:train_end]
        elif self.data_type == "val":
            self.data_input = self.data_input[val_start:val_end]
            self.data_mask = self.data_mask[val_start:val_end]
        elif self.data_type == "test":
            self.data_input = self.data_input[test_start:test_end]
            self.data_mask = self.data_mask[test_start:test_end]

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
