from copy import deepcopy

import paddle
import paddle.nn as nn

from dataset import SpatialGraph
from models import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    MultiHeadAttentionAwareTemporalContext,
    SpatialGraphNeuralNetwork,
    TrafficFlowEmbedding,
)


class STNXL(nn.Layer):
    def __init__(self, training_args, graph: SpatialGraph):
        super(STNXL, self).__init__()

        self.training_args = training_args
        self.graph = graph

        self.encoder_embedding = TrafficFlowEmbedding(args=training_args)
        self.decoder_embedding = TrafficFlowEmbedding(args=training_args)

        attn_ss = MultiHeadAttentionAwareTemporalContext(
            args=training_args, query_conv_type="1DConv", key_conv_type="1DConv"
        )
        attn_st = MultiHeadAttentionAwareTemporalContext(
            args=training_args, query_conv_type="causal", key_conv_type="1DConv"
        )
        attn_tt = MultiHeadAttentionAwareTemporalContext(
            args=training_args, query_conv_type="causal", key_conv_type="causal"
        )

        spatial_attention_gcn = SpatialGraphNeuralNetwork(
            args=training_args, graph=self.graph
        )

        encoderLayer = EncoderLayer(
            training_args.d_model,
            self_attn=deepcopy(attn_ss),
            gcn=deepcopy(spatial_attention_gcn),
        )
        decoderLayer = DecoderLayer(
            training_args.d_model,
            self_attn=deepcopy(attn_tt),
            cross_attn=deepcopy(attn_st),
            gcn=deepcopy(spatial_attention_gcn),
        )

        self.encoder = Encoder(encoderLayer, training_args.encoder_num_layers)
        self.decoder = Decoder(decoderLayer, training_args.decoder_num_layers)

        self.generator = nn.Linear(
            training_args.d_model, training_args.decoder_output_size
        )
        self.decoder_output = None

    def encode(self, src):
        src_dense = self.encoder_embedding(src)
        encoder_output = self.encoder(src_dense)
        return encoder_output

    def decode(self, encoder_output, tgt):
        tgt_dense = self.decoder_embedding(tgt)
        self.decoder_output = self.decoder(x=tgt_dense, memory=encoder_output)
        return self.generator(self.decoder_output)

    def update_graph(self):
        with paddle.no_grad():
            corr = paddle.einsum(
                "btnd,btmd->nm", self.decoder_output, self.decoder_output
            )
            values, indices = paddle.topk(
                corr, k=self.training_args.node_top_k, axis=-1
            )
            values = paddle.nn.functional.softmax(values, axis=-1)

            values, indices = values.numpy(), indices.numpy()
            edge_src, edge_dst, edge_weights = [], [], []
            node_nums = corr.shape[0]
            for i in range(node_nums):
                for j in range(self.training_args.node_top_k):
                    if indices[i, j] <= i:
                        edge_src.append(i)
                        edge_dst.append(indices[i, j])
                        edge_weights.append(values[i, j])
                # edge_src.extend([i] * self.training_args.node_top_k)
                # edge_dst.extend(list(indices[i]))
                # edge_weights.extend(list(values[i]))

            self.graph.edge_src_idx = deepcopy(edge_src)
            self.graph.edge_dst_idx = deepcopy(edge_dst)
            self.graph.edge_weights = deepcopy(edge_weights)
            self.graph.build_group_graph(n=2)
            if paddle.distributed.get_rank() == 0:
                self.graph.save_graph()
            self.apply(self.apply_new_graph)

    def apply_new_graph(self, layer):
        if isinstance(layer, SpatialGraphNeuralNetwork):
            layer.graph = deepcopy(self.graph)
            layer.group_nums = self.graph.group_mapping_node_nums[1][1]
            layer.edge_src_idx = paddle.to_tensor(self.graph.edge_src_idx)
            layer.edge_dst_idx = paddle.to_tensor(self.graph.edge_dst_idx)
            layer.down_sampling_edge_dst_idx = paddle.to_tensor(
                self.graph.group_mapping_edge_dst_idx[1]
            )
            layer.group_connect_edge_src_idx = paddle.to_tensor(
                self.graph.group_connect_edge_src_idx[1]
            )
            layer.group_connect_edge_dst_idx = paddle.to_tensor(
                self.graph.group_connect_edge_dst_idx[1]
            )
            layer.up_sampling_edge_dst_idx = paddle.to_tensor(
                self.graph.group_mapping_edge_dst_idx[1]
            )

    def forward(self, src, tgt):
        encoder_output = self.encode(src)
        output = self.decode(encoder_output, tgt)
        return output
