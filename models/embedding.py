import paddle
import paddle.nn as nn


class SpatialPositionalEmbedding(nn.Layer):
    def __init__(self, args):
        super(SpatialPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.embedding = paddle.nn.Embedding(args.num_nodes, args.d_model)

    def forward(self, x):
        """
        :param x: B,T,N,D
        :return: [1,1,N,D]
        """
        B, T, N, D = x.shape
        x_index = paddle.arange(N)
        embed = self.embedding(x_index).unsqueeze([0, 1])  # [N,D]->[1,1,N,D]
        return embed


class TemporalPositionalEmbedding(nn.Layer):
    def __init__(self, args, max_len):
        super(TemporalPositionalEmbedding, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)
        self.max_len = max_len
        self.d_model = args.d_model
        # computing the positional encodings once in log space
        pe = paddle.zeros([max_len, self.d_model])
        for pos in range(max_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = paddle.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = paddle.cos(
                    pos / (10000 ** ((2 * (i + 1)) / self.d_model))
                )

        pe = pe.unsqueeze([0, 2])  # [1,max_len,1,D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [B,T,N,D]

        Returns:[B,T,N,D]
        """
        lookup_index = paddle.arange(x.shape[1])
        embed = paddle.index_select(self.pe, lookup_index, axis=1)  # [1,T,1,D]
        return embed


class TrafficFlowEmbedding(nn.Layer):
    def __init__(self, args):
        super(TrafficFlowEmbedding).__init__()
        self.args = args

        self.dense = nn.Sequential(
            nn.Linear(self.args.input_size, self.args.d_model, bias_attr=True),
            nn.Silu(),
            nn.Linear(self.args.d_model, self.args.d_model, bias_attr=True),
        )

        self.spatial_position_embedding = SpatialPositionalEmbedding(args=args)
        self.temporal_position_embedding = TemporalPositionalEmbedding(args=args)

        self.layer_norm = nn.LayerNorm(self.args.d_model)

    def forward(self, x):
        x = self.dense(x)
        spatial_emb = self.spatial_position_embedding(x)
        temporal_emb = self.temporal_position_embedding(x)
        x = x + self.layer_norm(x + spatial_emb + temporal_emb)
        return x
