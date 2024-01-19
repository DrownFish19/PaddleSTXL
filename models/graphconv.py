import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pandas as pd
from geopy.distance import geodesic

from dataset.data_utils import haversine


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
        self.node_df = pd.read_csv(args.node_path)
        self.node_nums = len(self.node_df)

        if build:
            self.edge_src_idx = []
            self.edge_dst_idx = []
            self.edge_weights = []
            self.build_graph()
            self.save_graph()
        else:
            self.load_graph()

    def build_graph(self):
        """build graph according to the node dataframe"""
        self.node_nums = len(self.node_df)
        lon = self.node_df["lon"].values
        lat = self.node_df["lat"].values

        node_distances = {}
        for i in range(self.node_nums):
            # print(i, flush=True)
            row_i = self.node_df.iloc[i]
            # haversine method to calculate the distance
            distance = haversine(row_i["lon"], row_i["lat"], lon, lat)
            node_distances[i] = distance

        for id, distance in node_distances.items():
            if len(distance) == 0:
                continue
            topk_indices = np.argpartition(distance, self.args.node_top_k)
            topk_indices = topk_indices[: self.args.node_top_k]

            # build undirected graph. from node with small num to node with big num
            for k in topk_indices:
                if id <= k and distance[k] < self.args.node_max_dis:
                    self.edge_src_idx.append(id)
                    self.edge_dst_idx.append(k)
                    self.edge_weights.append(distance[k])

        # Normalize the weights
        self.edge_weights = [-w for w in self.edge_weights]
        max_weight = max(self.edge_weights)
        min_weight = min(self.edge_weights)
        self.edge_weights = [
            (w - min_weight) * 0.9 / (max_weight - min_weight) + 0.1
            for w in self.edge_weights
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


class SpatialGraphMLP(nn.Layer):
    def __init__(
        self, in_features, out_features, latent_features=None, layer_norm=True
    ):
        super().__init__()

        if latent_features is None:
            latent_features = out_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, latent_features, bias_attr=True),
            nn.Silu(),
            nn.Linear(latent_features, out_features, bias_attr=True),
        )
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, feat):
        if self.layer_norm:
            out = self.layer_norm(self.mlp(feat))
        else:
            out = self.mlp(feat)
        return out


class SpatialGraphNeuralNetwork(nn.Layer):
    def __init__(self, args, graph: GraphST):
        super().__init__()
        self.args = args
        self.edge_in_dim = args.d_model * 2
        self.edge_out_dim = args.d_model
        self.node_in_dim = args.d_model * 2
        self.node_out_dim = args.d_model

        self.graph = graph
        self.node_nums = self.graph.node_nums
        self.edge_src_idx = paddle.to_tensor(self.graph.edge_src_idx)
        self.edge_dst_idx = paddle.to_tensor(self.graph.edge_dst_idx)

        self.edge_layer = SpatialGraphMLP(self.edge_in_dim, self.edge_out_dim)
        self.node_layer = SpatialGraphMLP(self.node_in_dim, self.node_out_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (paddle.Tensor): [B,T,N,D]

        Returns:
            _type_: _description_
        """
        # 更新edge特征
        # [B,T,E,D]
        src_feat = paddle.gather(x, self.edge_src_idx, axis=2)
        dst_feat = paddle.gather(x, self.edge_dst_idx, axis=2)

        # [B,T,E,D*2]
        edge_feat = paddle.concat([src_feat, dst_feat], axis=-1)
        edge_feats_out = self.edge_layer(edge_feat)  # [B,T,E,D]
        B, T, E, D = edge_feats_out.shape

        # [B,T,E,D] -> [E,B,T,D]
        edge_feats_out = edge_feats_out.transpose([2, 0, 1, 3])
        # 更新node特征
        edge_feats_scatter_src = paddle.zeros([self.node_nums, B, T, D])
        edge_feats_scatter_dst = paddle.zeros([self.node_nums, B, T, D])
        node_feats_concat = paddle.concat(
            [
                paddle.scatter(
                    edge_feats_scatter_src,
                    self.edge_src_idx,
                    edge_feats_out,
                    overwrite=False,
                ),
                paddle.scatter(
                    edge_feats_scatter_dst,
                    self.edge_dst_idx,
                    edge_feats_out,
                    overwrite=False,
                ),
            ],
            axis=-1,
        )
        node_feats_out = self.node_layer(node_feats_concat)  # [N,B,T,D]
        node_feats_out = node_feats_out.transpose([1, 2, 0, 3])

        return x + node_feats_out


class SpatialAttentionLayer(nn.Layer):
    """
    compute spatial attention scores
    """

    def __init__(self, dropout=0.0):
        super(SpatialAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: (B, T, E, D)
        :return: (B, T, N, N)
        """
        B, T, E, D = x.shape

        x = x.transpose([0, 2, 1, 3])  # [B,T,N,F_in]
        # [B,T,N,F_in][B,T,F_in,N]=[B*T,N,N]
        score = paddle.matmul(x, x, transpose_y=True) / math.sqrt(D)
        score = self.dropout(F.softmax(score, axis=-1))  # [B,T,N,N]
        return score
