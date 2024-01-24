import paddle.nn as nn

from .embedding import TrafficFlowEmbedding


class STLSTM(nn.Layer):
    def __init__(self, training_args):
        super(STLSTM, self).__init__()

        self.training_args = training_args
        self.embedding = TrafficFlowEmbedding(args=training_args)
        self.lstm = nn.LSTM(
            self.training_args.d_model, self.training_args.d_model, 4, time_major=False
        )
        self.norm = nn.LayerNorm(self.training_args.d_model)
        self.generator = nn.Linear(
            training_args.d_model, training_args.decoder_output_size
        )

    def encode(self, src):
        src_dense = self.embedding(src)  # B,T,N,D
        B, T, N, D = src_dense.shape
        src_dense = src_dense.transpose([0, 2, 1, 3]).reshape([B * N, T, D])
        encoder_output, _ = self.lstm(src_dense)
        encoder_output = self.norm(encoder_output)
        encoder_output = encoder_output.reshape([B, N, T, D]).transpose([0, 2, 1, 3])
        return encoder_output

    def decode(self, encoder_output, tgt):
        return self.generator(encoder_output)

    def forward(self, src, tgt):
        encoder_output = self.encode(src)
        output = self.decode(encoder_output, tgt)
        return output
