import argparse
import json

parser = argparse.ArgumentParser(description="Traffic Flow Forecasting")

# data config
parser.add_argument("--dataset_name", type=str, default="PEMS", help="dataset name")
parser.add_argument("--whole_data_path", type=str, default="data/flow.npz")
parser.add_argument("--train_data_path", type=str, default="data/train.npz")
parser.add_argument("--val_data_path", type=str, default="data/val.npz")
parser.add_argument("--test_data_path", type=str, default="data/test.npz")
parser.add_argument("--scaler_data_path", type=str, default="data/scaler.pkl")
parser.add_argument("--node_path", type=str, default="")
# 如果需要load指定epoch的参数时，需要将此处的adj_path设置为保存的图数据路径
parser.add_argument("--adj_path", type=str, default="data/adj_weights.csv")
# for save
parser.add_argument("--group_path", type=str, default="data/group_weights.csv")
# for save
parser.add_argument("--mapping_path", type=str, default="data/mapping_weights.csv")
parser.add_argument("--split", type=str, default="6:2:2", help="data split")
parser.add_argument("--scale", type=bool, default=True, help="data norm scale")
parser.add_argument("--num_nodes", type=int, default=19875)
parser.add_argument("--node_top_k", type=int, default=10)
parser.add_argument("--node_max_dis", type=int, default=30)

# model config
parser.add_argument("--model_name", type=str, default="PaddleSTXL", help="model name")
parser.add_argument("--his_len", type=int, default=12, help="history data length")
parser.add_argument("--tgt_len", type=int, default=12, help="tgt data length")
parser.add_argument("--input_size", type=int, default=1)
parser.add_argument("--decoder_output_size", type=int, default=1)
parser.add_argument("--encoder_num_layers", type=int, default=2)
parser.add_argument("--decoder_num_layers", type=int, default=1)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--attention", type=str, default="Corr", help="Corr,Vanilla")
parser.add_argument("--head", type=int, default=4, help="head")
parser.add_argument("--kernel_size", type=int, default=3, help="kernel_size")
parser.add_argument("--smooth_layer_num", type=int, default=1)

# train config
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-8)
parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
parser.add_argument("--train_epochs", type=int, default=150, help="train epochs")
parser.add_argument("--eval_interval_epochs", type=int, default=1, help="train epochs")
parser.add_argument("--update_graph_epochs", type=int, default=2, help="")
parser.add_argument("--finetune_epochs", type=int, default=50, help="finetune epochs")
parser.add_argument("--warmup_step", type=int, default=50, help="warmup_step")
parser.add_argument("--decay_step", type=int, default=100, help="decay_step")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--patience", type=int, default=8, help="early stopping patience")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
parser.add_argument("--continue_training", type=bool, default=False, help="")
parser.add_argument("--tqdm", type=bool, default=False, help="")
parser.add_argument("--fp16", type=bool, default=True, help="")
parser.add_argument("--distribute", type=bool, default=True, help="")


def get_args_from_json(json_file_path, args_obj):
    with open(json_file_path) as f:
        json_dict = json.load(fp=f)

    for key in json_dict.keys():
        setattr(args_obj, key, json_dict[key])

    return args_obj


args_obj = parser.parse_args()
args = get_args_from_json("configs/PEMS.json", args_obj)
# args = get_args_from_json("configs/PEMS03.json", args_obj)
# args = get_args_from_json("configs/PEMS04.json", args_obj)
# args = get_args_from_json("configs/PEMS07.json", args_obj)
# args = get_args_from_json("configs/PEMS08.json", args_obj)
# args = get_args_from_json("configs/HZME_INFLOW.json", args_obj)
# args = get_args_from_json("configs/HZME_OUTFLOW.json", args_obj)
