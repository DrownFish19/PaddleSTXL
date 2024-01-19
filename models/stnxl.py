from copy import deepcopy

import paddle.nn as nn
from attention import MultiHeadAttentionAwareTemporalContext
from embedding import TrafficFlowEmbedding
from endecoder import Decoder, DecoderLayer, Encoder, EncoderLayer
from graphconv import SpatialGraphNeuralNetwork


class STNXL(nn.Layer):
    def __init__(self, training_args, graph):
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

    def encode(self, src):
        src_dense = self.encoder_embedding(src)
        encoder_output = self.encoder(src_dense)
        return encoder_output

    def decode(self, encoder_output, tgt):
        tgt_dense = self.decoder_embedding(tgt)
        decoder_output = self.decoder(x=tgt_dense, memory=encoder_output)
        return self.generator(decoder_output)

    def forward(self, src, tgt):
        encoder_output = self.encode(src)
        output = self.decode(encoder_output, tgt)
        return output
