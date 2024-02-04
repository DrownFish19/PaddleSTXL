import paddle
import paddle.nn as nn

from dataset import SpatialGraph


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
    def __init__(self, args, graph: SpatialGraph):
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
        return node_feats_out + up_sampling_node_feats
