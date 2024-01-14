import math
import multiprocessing

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pandas as pd
import tqdm
from geopy.distance import geodesic


def get_distance(row_i, row_j):
    dis = geodesic(
        (row_i["lat"], row_i["lon"]),
        (row_j["lat"], row_j["lon"]),
    ).kilometers

    return row_i["ID"], row_j["ID"], dis


class GraphST:
    def __init__(self, args, build=True):
        """_summary_

        Args:
            args (_type_): args
            node_df (pd.DataFrame): the node dataframe is a csv file
                                        with the following format:
            ID,dist,lat,lon
            308511,3,38.761062,-120.569835
            308512,3,38.761182,-120.569866
            311831,3,38.409253,-121.48412
            311832,3,38.409782,-121.48468
        """
        self.args = args
        if build:
            self.node_df = pd.read_csv(args.node_path)
            self.edge_src_idx = []
            self.edge_dst_idx = []
            self.edge_weights = []
            self.build_graph()
            self.save_graph()
        else:
            self.load_graph()

    def build_graph(self):
        """build graph according to the node dataframe"""
        pool = multiprocessing.Pool(processes=30)
        self.node_nums = len(self.node_df)
        node_distances = {id: [] for id in self.node_df["ID"]}
        results = []
        for i in range(self.node_nums):
            print(i)
            row_i = self.node_df.iloc[i]
            for j in range(i + 1, self.node_nums - 1):
                row_j = self.node_df.iloc[j]
                results.append(pool.apply_async(get_distance, args=(row_i, row_j)))
        for res in tqdm.tqdm(results):
            row_i_id, row_j_id, dist = res.get()
            node_distances[row_i_id].append((row_j_id, dist))
            node_distances[row_j_id].append((row_i_id, dist))
            print(row_i_id, row_j_id, dist)

        for id, dist_list in node_distances.items():
            if len(dist_list) == 0:
                continue
            dist_list.sort(key=lambda x: x[1])
            need_add = [dist_list[0][0]]  # id
            need_weights = [dist_list[0][1]]  # dist
            for k in range(1, min(self.args.node_top_k, len(dist_list))):
                if dist_list[k][1] < self.args.node_max_dis:
                    need_add.append(dist_list[k][0])
                    need_weights.append(dist_list[k][1])

            self.edge_src_idx.extend([id] * len(need_add))
            self.edge_dst_idx.extend(need_add)
            self.edge_weights.extend(need_weights)

        self.edge_weights = [1 / w if w != 0 else 0 for w in self.edge_weights]
        # Normalize the weights
        min_weight = min(self.edge_weights)
        max_weight = max(self.edge_weights)
        self.edge_weights = [
            (w - min_weight) / (max_weight - min_weight) for w in self.edge_weights
        ]

    def load_graph(self):
        dataframe = pd.read_csv(self.args.adj_path)
        self.edge_src_idx = dataframe["src"].values.tolist()
        self.edge_dst_idx = dataframe["dst"].values.tolist()
        self.edge_weights = dataframe["weight"].values.tolist()

    def save_graph(self):
        dataframe = pd.DataFrame(
            {
                "src": self.edge_src_idx,
                "dst": self.edge_dst_idx,
                "weight": self.edge_weights,
            }
        )
        dataframe.to_csv(self.args.adj_path, index=False)


class GCN(nn.Layer):
    def __init__(self, training_args, d_model, norm_adj_matrix, norm_sc_matrix):
        super(GCN, self).__init__()
        self.norm_adj_matrix = norm_adj_matrix
        self.norm_sc_matrix = norm_sc_matrix
        self.Theta = nn.Linear(
            d_model,
            d_model,
            bias_attr=False,
        )
        self.alpha = paddle.create_parameter(
            shape=[1],
            dtype=paddle.get_default_dtype(),
        )
        if training_args.no_adj:
            self.alpha.set_value(paddle.to_tensor([0.0]))
            self.alpha.stop_gradient = True
        self.beta = paddle.create_parameter(shape=[1], dtype=paddle.get_default_dtype())

    def forward(self, x):
        """
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        """
        adj = paddle.add(
            self.alpha * self.norm_adj_matrix,
            self.beta * self.norm_sc_matrix,
        )
        x_gcn = paddle.matmul(adj, x)
        # [N,N][B,N,in]->[B,N,in]->[B,N,out]
        return F.relu(self.Theta(x_gcn))


class SpatialAttentionLayer(nn.Layer):
    """
    compute spatial attention scores
    """

    def __init__(self, dropout=0.0):
        super(SpatialAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: (B, N, T, D)
        :return: (B, T, N, N)
        """
        B, N, T, D = x.shape
        x = x.transpose([0, 2, 1, 3])  # [B,T,N,F_in]
        # [B,T,N,F_in][B,T,F_in,N]=[B*T,N,N]
        score = paddle.matmul(x, x, transpose_y=True) / math.sqrt(D)
        score = self.dropout(F.softmax(score, axis=-1))  # [B,T,N,N]
        return score


class SpatialAttentionGCN(nn.Layer):
    def __init__(self, args, adj_matrix, sc_matrix, is_scale=True):
        super(SpatialAttentionGCN, self).__init__()
        self.norm_adj = adj_matrix
        self.norm_sc = sc_matrix
        self.args = args
        self.linear = nn.Linear(args.d_model, args.d_model, bias_attr=False)
        self.is_scale = is_scale
        self.SAt = SpatialAttentionLayer(dropout=args.dropout)
        self.alpha = paddle.create_parameter(
            shape=[1], dtype=paddle.get_default_dtype()
        )
        if args.no_adj:
            self.alpha.set_value(paddle.to_tensor([0.0]))
            self.alpha.stop_gradient = True
        self.beta = paddle.create_parameter(shape=[1], dtype=paddle.get_default_dtype())

    def forward(self, x):
        """
        spatial graph convolution
        :param x: (B, N, T, F_in)
        :return: (B, N, T, F_out)
        """
        B, N, T, D = x.shape
        spatial_attention = self.SAt(x)  # [B, T, N, N]
        if self.is_scale:
            spatial_attention = spatial_attention / math.sqrt(self.args.d_model)
        x = x.transpose([0, 2, 1, 3])  # [B,T,N,D]

        adj = paddle.add(
            self.alpha * paddle.multiply(spatial_attention, self.norm_adj),
            self.beta * paddle.multiply(spatial_attention, self.norm_sc),
        )
        x_gcn = paddle.matmul(adj, x)
        # [B, N, T, D]
        return F.relu(self.linear(x_gcn).transpose([0, 2, 1, 3]))