import numpy as np

from args import args


def split_train_val_test_dataset(args):
    # step 1: load data
    # [T, N, D]
    whole_data = np.load(args.whole_data_path)["data"]
    seq_len, num_nodes, dims = whole_data.shape
    print("whole data shape:", whole_data.shape, flush=True)

    mask = np.ones((seq_len, num_nodes, 1))
    if "HZME" in args.whole_data_path:
        # add data mask
        mask[np.arange(seq_len) % 288 < 6 * 12, :, :] = 0
    whole_data = np.concatenate([mask, whole_data], axis=-1)
    print("whole data (with data mask) shape:", whole_data.shape, flush=True)
    # step 2: compute data ratio
    train_ratio, val_ratio, test_ratio = map(int, args.split.split(":"))
    sum_ratio = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = (
        train_ratio / sum_ratio,
        val_ratio / sum_ratio,
        test_ratio / sum_ratio,
    )

    train_size, val_size, test_size = (
        int(seq_len * train_ratio),
        int(seq_len * val_ratio),
        int(seq_len * test_ratio),
    )
    train_start, train_end = 0, train_size
    val_start, val_end = train_end, train_end + val_size
    test_start, test_end = val_end, min(val_end + test_size, seq_len)
    # step 3: save data
    np.savez_compressed(args.train_data_path, data=whole_data[train_start:train_end])
    np.savez_compressed(args.val_data_path, data=whole_data[val_start:val_end])
    np.savez_compressed(args.test_data_path, data=whole_data[test_start:test_end])


if __name__ == "__main__":
    split_train_val_test_dataset(args)
