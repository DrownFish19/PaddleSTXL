import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

from dataset.data_utils import haversine

try:
    import hssinfo
except:
    pass


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

        self.build_laplacian_matrix()
        chebyshev_matrix = self.build_chebyshev_polynomials(k=1)
        self.chebyshev_matrix = chebyshev_matrix[-1]

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

    def build_laplacian_matrix(self):
        # self.matrix_sparse = sparse.csc_matrix(
        #     (self.edge_weights, (self.edge_src_idx, self.edge_dst_idx)),
        #     shape=(self.node_nums, self.node_nums),
        # )
        self.matrix_sparse = sparse.csc_matrix(
            (
                np.ones_like(np.array(self.edge_weights)),
                (self.edge_src_idx, self.edge_dst_idx),
            ),
            shape=(self.node_nums, self.node_nums),
        )
        self.matrix_sparse = self.matrix_sparse + self.matrix_sparse.T  # 转换为无向图结构
        id = sparse.identity(self.node_nums, format="csc")
        self.matrix_sparse = self.matrix_sparse - id  # 相加过程中自权重翻倍，减去自权重

        # A_{sym} = I - D^{-0.5} * A * D^{-0.5}
        row_sum = self.matrix_sparse.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.0
        row_sum_inv_sqrt[np.isnan(row_sum_inv_sqrt)] = 0.0
        deg_inv_sqrt = sparse.diags(row_sum_inv_sqrt, format="csc")
        sym_norm_adj = deg_inv_sqrt.dot(self.matrix_sparse).dot(deg_inv_sqrt)

        self.laplacian_matrix = id - sym_norm_adj

    def build_chebyshev_polynomials(self, k):
        """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
        if getattr(self, "laplacian_matrix", None) is None:
            self.build_laplacian_matrix()

        degree_diags_matrix = sparse.identity(self.node_nums, format="csc")
        max_eigen_val = linalg.norm(self.laplacian_matrix, 2)

        # Calculate Chebyshev polynomials
        scaled_L = (2.0 / max_eigen_val) * self.laplacian_matrix - degree_diags_matrix
        T_k = [degree_diags_matrix, scaled_L]

        for i in range(2, k + 1):
            T_k.append(2 * scaled_L * T_k[i - 1] - T_k[i - 2])

        return T_k

    def get_cluster_group(self, node_nums, edge_src_idx, edge_dst_idx, edge_weights):
        dist.init_parallel_env()
        res = paddle.zeros([node_nums], dtype=paddle.int32)
        if paddle.distributed.get_rank() == 0:
            res = hssinfo.cluster(
                paddle.arange(node_nums),
                paddle.to_tensor(edge_src_idx, dtype=paddle.int32),
                paddle.to_tensor(edge_dst_idx, dtype=paddle.int32),
                paddle.to_tensor(edge_weights, dtype=paddle.float32),
            )
            expected_place = paddle.framework._current_expected_place()
            res = res._copy_to(expected_place, False)
        dist.barrier()
        dist.broadcast(res, src=0)
        res = res.numpy()
        return res

    def build_group_graph(self, n):
        assert n >= 2, "group levels should be larger than or equal to 2"
        # 分组划分映射，二部图
        self.group_mapping_node_nums = {}
        self.group_mapping_edge_src_idx = {i: [] for i in range(n)}
        self.group_mapping_edge_dst_idx = {i: [] for i in range(n)}

        self.group_connect_edge_src_idx = {}
        self.group_connect_edge_dst_idx = {}
        self.group_connect_edge_weights = {}

        res = self.get_cluster_group(
            self.node_nums, self.edge_src_idx, self.edge_dst_idx, self.edge_weights
        )

        self.group_mapping_edge_src_idx[1] = [idx for idx in range(self.node_nums)]
        self.group_mapping_edge_dst_idx[1] = list(res)
        self.group_mapping_node_nums[1] = (self.node_nums, max(res) + 1)

        conn_src, conn_dst, conn_weights = self._build_group_edge(
            res,
            edge_src_idx=self.edge_src_idx,
            edge_dst_idx=self.edge_dst_idx,
            edge_weights=self.edge_weights,
        )
        self.group_connect_edge_src_idx[1] = conn_src
        self.group_connect_edge_dst_idx[1] = conn_dst
        self.group_connect_edge_weights[1] = conn_weights

        for i in range(2, n):
            last_group_nums = self.group_mapping_node_nums[i - 1][1]
            res = self.get_cluster_group(
                last_group_nums, conn_src, conn_dst, conn_weights
            )
            if max(res) + 1 == last_group_nums:
                # without re-grouping
                break

            self.group_mapping_edge_src_idx[i] = [idx for idx in range(last_group_nums)]
            self.group_mapping_edge_dst_idx[i] = list(res)
            self.group_mapping_node_nums[i] = (last_group_nums, max(res) + 1)

            conn_src, conn_dst, conn_weights = self._build_group_edge(
                res,
                edge_src_idx=conn_src,
                edge_dst_idx=conn_dst,
                edge_weights=conn_weights,
            )
            self.group_connect_edge_src_idx[i] = conn_src
            self.group_connect_edge_dst_idx[i] = conn_dst
            self.group_connect_edge_weights[i] = conn_weights

    def _build_group_edge(
        self, res: np.ndarray, edge_src_idx, edge_dst_idx, edge_weights
    ):
        res_dict = {}  # id => group
        weights_dict = {}
        edges_dict = []
        for i in range(len(res)):
            res_dict[i] = res[i]

        for edge_idx in range(len(edge_src_idx)):
            src_idx, dst_idx = edge_src_idx[edge_idx], edge_dst_idx[edge_idx]
            src_group, dst_group = res_dict[src_idx], res_dict[dst_idx]
            if src_group != dst_group:
                key = (src_group, dst_group)
                if key not in edges_dict:
                    edges_dict.append(key)
                    weights_dict[key] = edge_weights[edge_idx]
                else:
                    weights_dict[key] += edge_weights[edge_idx]

        connect_edge_src_idx = []
        connect_edge_dst_idx = []
        connect_edge_weights = []
        for key in weights_dict:
            src_group, dst_group = key
            connect_edge_src_idx.append(src_group)
            connect_edge_dst_idx.append(dst_group)
            connect_edge_weights.append(weights_dict[key])
        return connect_edge_src_idx, connect_edge_dst_idx, connect_edge_weights

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
        self.group_nums = self.graph.group_mapping_node_nums[1][1]
        self.edge_src_idx = paddle.to_tensor(self.graph.edge_src_idx)
        self.edge_dst_idx = paddle.to_tensor(self.graph.edge_dst_idx)
        self.down_sampling_edge_dst_idx = paddle.to_tensor(
            self.graph.group_mapping_edge_dst_idx[1]
        )
        self.group_connect_edge_src_idx = paddle.to_tensor(
            self.graph.group_connect_edge_src_idx[1]
        )
        self.group_connect_edge_dst_idx = paddle.to_tensor(
            self.graph.group_connect_edge_dst_idx[1]
        )
        self.up_sampling_edge_dst_idx = paddle.to_tensor(
            self.graph.group_mapping_edge_dst_idx[1]
        )

        self.edge_layer = SpatialGraphMLP(self.edge_in_dim, self.edge_out_dim)
        self.node_layer = SpatialGraphMLP(self.node_in_dim, self.node_out_dim)
        self.node2group_layer = SpatialGraphMLP(self.node_out_dim, self.node_out_dim)
        self.group2node_layer = SpatialGraphMLP(self.node_out_dim, self.node_out_dim)
        self.group_layer = SpatialGraphMLP(self.node_in_dim, self.node_out_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (paddle.Tensor): [B,T,N,D]

        Returns:
            _type_: _description_
        """
        # 1. 更新edge特征
        src_feat = paddle.gather(x, self.edge_src_idx, axis=2)  # [B,T,E,D]
        dst_feat = paddle.gather(x, self.edge_dst_idx, axis=2)  # [B,T,E,D]
        edge_feat = paddle.concat([src_feat, dst_feat], axis=-1)  # [B,T,E,D*2]

        # 2. 计算edge特征的相似度
        # [B,N,E,1,D] * [B,N,E,D,1] => [B,N,E,1]
        edge_attention = paddle.matmul(edge_feat.unsqueeze(-2), edge_feat.unsqueeze(-1))
        edge_attention = edge_attention.squeeze(-1)
        edge_feats_out = self.edge_layer(edge_feat)  # [B,T,E,D]
        edge_feats_out = edge_attention * edge_feats_out

        # 3. 更新node特征
        B, T, E, D = edge_feats_out.shape

        # [B,T,E,D] -> [E,B,T,D]
        edge_feats_out = edge_feats_out.transpose([2, 0, 1, 3])
        # 3.1. 节点间的edge特征聚合
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
        )  # [N,B,T,D*2]
        node_feats_out = self.node_layer(node_feats_concat)  # [N,B,T,D]

        # 3.2. 节点-组的edge特征聚合
        down_sampling_node_feats = paddle.zeros([self.group_nums, B, T, D])
        down_sampling_node_feats = paddle.scatter(
            down_sampling_node_feats,
            self.down_sampling_edge_dst_idx,
            node_feats_out,
            overwrite=False,
        )  # [G,B,T,D]
        down_sampling_node_feats = self.node2group_layer(down_sampling_node_feats)

        # 3.3. 组间的edge特征聚合
        group_connect_node_feats_src = paddle.zeros([self.group_nums, B, T, D])
        group_connect_node_feats_dst = paddle.zeros([self.group_nums, B, T, D])
        group_connect_node_feats = paddle.concat(
            [
                paddle.scatter(
                    group_connect_node_feats_src,
                    self.group_connect_edge_src_idx,
                    paddle.gather(
                        down_sampling_node_feats,
                        self.group_connect_edge_dst_idx,
                        axis=0,
                    ),
                    overwrite=False,
                ),
                paddle.scatter(
                    group_connect_node_feats_dst,
                    self.group_connect_edge_dst_idx,
                    paddle.gather(
                        down_sampling_node_feats,
                        self.group_connect_edge_src_idx,
                        axis=0,
                    ),
                    overwrite=False,
                ),
            ],
            axis=-1,
        )  # [G,B,T,D*2]
        # [G,B,T,D]
        group_connect_node_feats = self.group_layer(group_connect_node_feats)

        # 3.4. 组-节点的edge特征聚合
        # [N,B,T,D]
        up_sampling_node_feats = paddle.gather(
            group_connect_node_feats, self.up_sampling_edge_dst_idx, axis=0
        )
        up_sampling_node_feats = self.group2node_layer(up_sampling_node_feats)

        node_feats_out = node_feats_out.transpose([1, 2, 0, 3])
        up_sampling_node_feats = up_sampling_node_feats.transpose([1, 2, 0, 3])
        return x + node_feats_out + up_sampling_node_feats
