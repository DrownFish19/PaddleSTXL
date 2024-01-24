import paddle
import paddle.nn as nn
import paddle.nn.functional as F


from utils import clones


class STGCN(nn.Layer):
    def __init__(self, args, graph):
        super().__init__()
        self.args = args
        self.graph = graph

        self.st_blocks = clones(STConvBlock(self.args, graph), self.args.decoder_num_layers)
        self.output = OutputBlock(self.args)

    def forward(self, src, tgt):
        x = src
        for st_block in self.st_blocks:
            x = st_block(x)
        x = self.output(x)
        return x


class Align(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.align_conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), data_format="NHWC")

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): B,T,N,D

        Returns:
            _type_: _description_
        """
        if self.in_channels > self.out_channels:
            x = self.align_conv(x)
        elif self.in_channels < self.out_channels:
            B, T, N, D = x.shape
            x = paddle.concat([x, paddle.zeros([B, T, N, self.out_channels - self.in_channels])], axis=-1)
        else:
            x = x

        return x


class TemporalConvLayer(nn.Layer):

    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * residual connection *
    #        |                                |
    #        |    |--->--- casualconv2d ----- + -------|
    # -------|----|                                   ⊙ ------>
    #             |--->--- casualconv2d --- sigmoid ---|
    #

    # param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, args, c_in, c_out):
        super(TemporalConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)

        self.padding_causal = args.kernel_size - 1
        self.causal_conv = nn.Conv2D(
            args.d_model,
            args.d_model,
            (args.kernel_size, 1),
            padding=(self.padding_causal, 0),
            bias_attr=True,
            data_format="NHWC",
        )

    def forward(self, x):
        x_in = self.align(x)
        x_causal_conv = self.causal_conv(x_in)[:, :-self.padding_causal, : , :]
        x = F.relu(x_causal_conv + x_in)
        return x


class ChebyshevGraphConv(nn.Layer):
    def __init__(self, c_in, c_out, chebyshev_matrix):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.chebyshev_matrix = chebyshev_matrix
        self.weight = paddle.create_parameter(shape=[3, c_in, c_out], dtype=paddle.get_default_dtype())
        self.bias = paddle.create_parameter(shape=[c_out], dtype=paddle.get_default_dtype())

    def forward(self, x):
        # B, T, N, D = x.shape

        x_0 = x
        x_1 = paddle.einsum('hn,btnd->bthd', self.chebyshev_matrix, x)
        x_2 = paddle.einsum('hn,btnd->bthd', 2 * self.chebyshev_matrix, x_1) - x_0

        x = paddle.stack([x_0, x_1, x_2], dim=3)

        chebyshev_graph_conv = paddle.einsum('btknd,kdj->btnj', x, self.weight)
        chebyshev_graph_conv = chebyshev_graph_conv + self.bias
        return chebyshev_graph_conv


class GraphConv(nn.Layer):
    def __init__(self, c_in, c_out, laplacian_matrix):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.laplacian_matrix = laplacian_matrix

        self.weight = paddle.create_parameter(shape=[c_in, c_out], dtype=paddle.get_default_dtype())
        self.bias = paddle.create_parameter(shape=[c_out], dtype=paddle.get_default_dtype())

    def forward(self, x):
        # B, T, N, D = x.shape

        x = paddle.einsum('hi,btij->bthj', self.gso, x)
        graph_conv = paddle.einsum('bthi,ij->bthj', x, self.weight)
        return graph_conv + self.bias


class GraphConvLayer(nn.Layer):
    def __init__(self, c_in, c_out, graph):
        super(GraphConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.graph_conv = ChebyshevGraphConv(c_out, c_out, graph.chebyshev_matrix)
        # self.graph_conv = GraphConv(c_out, c_out, graph.laplacian_matrix)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): B,T,N,D

        Returns:
            _type_: _description_
        """
        x_gc_in = self.align(x)
        x_gc = self.graph_conv(x_gc_in)
        # x_gc = self.graph_conv(x_gc_in)
        x_gc_out = x_gc + x_gc_in

        return x_gc_out


class STConvBlock(nn.Layer):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, args, graph):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(args, args.input_size, args.d_model)
        self.graph_conv = GraphConvLayer(args.d_model, args.d_model, graph)
        self.tmp_conv2 = TemporalConvLayer(args, args.d_model, args.d_model)
        self.tc2_ln = nn.LayerNorm(args.d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x)
        return x


class OutputBlock(nn.Layer):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(args, args.d_model, args.d_model)
        self.fc1 = nn.Linear(in_features=args.d_model, out_features=args.d_model)
        self.fc2 = nn.Linear(in_features=args.d_model, out_features=args.decoder_output_size)
        self.tc1_ln = nn.LayerNorm(args.d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
