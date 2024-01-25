import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ASTGCN(nn.Layer):
    def __init__(
        self,
        args,
        graph,
    ):
        super().__init__()

        self.BlockList = nn.LayerList([ASTGCN_block(args, args.input_size, graph)])
        self.BlockList.extend(
            [
                ASTGCN_block(args, args.d_model, graph)
                for _ in range(args.decoder_num_layers - 1)
            ]
        )

        self.final_conv = nn.Conv2D(
            args.his_len,
            args.tgt_len,
            kernel_size=(1, args.d_model),
            bias_attr=True,
            data_format="NCHW",
        )

    def forward(self, src, tgt):
        """_summary_

        Args:
            src (_type_): [B,T,N,D]
            tgt (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = src
        for block in self.BlockList:
            x = block(x)
        # x : [B,T,N,F] => [B,T_out,N,1]
        output = self.final_conv(x)
        return output


class TemporalAttentionLayer(nn.Layer):
    def __init__(self, in_channels, vertices, timesteps):
        super().__init__()
        dtype = paddle.get_default_dtype()
        self.U1 = paddle.create_parameter(shape=[vertices], dtype=dtype)
        self.U2 = paddle.create_parameter(shape=[in_channels, vertices], dtype=dtype)
        self.U3 = paddle.create_parameter(shape=[in_channels], dtype=dtype)
        self.be = paddle.create_parameter(shape=[1, timesteps, timesteps], dtype=dtype)
        self.Ve = paddle.create_parameter(shape=[timesteps, timesteps], dtype=dtype)

    def forward(self, x):
        # [N]*[B,T,N,D] => [B,T,D]
        lhs = paddle.einsum("btd,dn-> btn", paddle.matmul(self.U1, x), self.U2)
        rhs = paddle.einsum("btnd,d-> bnt", x, self.U3)
        product = paddle.matmul(lhs, rhs)  # [B,T,N]*[B,N,T] => [B,T,T]
        E = paddle.matmul(self.Ve, F.sigmoid(product + self.be))
        E_normalized = F.softmax(E, axis=-1)

        return E_normalized


class SpatialAttentionLayer(nn.Layer):
    def __init__(self, in_channels, vertices, timesteps):
        super().__init__()
        dtype = paddle.get_default_dtype()
        self.W1 = paddle.create_parameter(shape=[timesteps], dtype=dtype)
        self.W2 = paddle.create_parameter(shape=[in_channels, timesteps], dtype=dtype)
        self.W3 = paddle.create_parameter(shape=[in_channels], dtype=dtype)
        self.bs = paddle.create_parameter(shape=[1, vertices, vertices], dtype=dtype)
        self.Vs = paddle.create_parameter(shape=[vertices, vertices], dtype=dtype)

    def forward(self, x):
        """
        x:[B,T,N,D]
        return: [B,N,N]
        """
        lhs = paddle.einsum("btnd,t -> bnd", x, self.W1)
        lhs = paddle.einsum("bnd,dt -> bnt", lhs, self.W2)
        rhs = paddle.matmul(x, self.W3)  # [B,T,N,D]*[D] => [B,T,N]
        product = paddle.matmul(lhs, rhs)
        S = paddle.matmul(self.Vs, F.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, axis=-1)
        return S_normalized  # [B, N, N]


class ChebyshevAttentionConvLayer(nn.Layer):
    def __init__(self, graph, in_channels, out_channels):
        super().__init__()
        dtype = paddle.get_default_dtype()
        self.K = len(graph.chebyshev_matrix)
        self.chebyshev_matrix = [
            paddle.to_tensor(i.toarray(), dtype=dtype) for i in graph.chebyshev_matrix
        ]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = paddle.create_parameter(
            shape=[self.K, in_channels, out_channels], dtype=dtype
        )

    def forward(self, x, spatial_attention):
        B, T, N, D = x.shape
        outputs = []
        for time_step in range(T):
            graph_signal = x[:, time_step, :, :]  # (b, N, F_in)
            output = paddle.zeros([B, N, self.out_channels])  # (b, N, F_out)

            for k in range(self.K):
                T_k_with_at = paddle.multiply(
                    spatial_attention, self.chebyshev_matrix[k]
                )
                # [B,N,N] * [B,N,D] => [B,N,D]
                rhs = paddle.matmul(T_k_with_at, graph_signal)
                # [B,N,D] * [D,D1] => [B,N,D1]
                output = output + paddle.matmul(rhs, self.Theta[k])
            outputs.append(output)  # [B, N, D]
        return F.relu(paddle.stack(outputs, axis=1))  # [B,T,N,D]


class ASTGCN_block(nn.Layer):
    def __init__(self, args, in_channels, graph):
        super(ASTGCN_block, self).__init__()
        self.TAt = TemporalAttentionLayer(in_channels, args.num_nodes, args.his_len)
        self.SAt = SpatialAttentionLayer(in_channels, args.num_nodes, args.his_len)
        self.chebyAtt = ChebyshevAttentionConvLayer(graph, in_channels, args.d_model)
        self.time_conv = nn.Conv2D(
            args.d_model,
            args.d_model,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias_attr=True,
            data_format="NHWC",
        )
        self.residual_conv = nn.Conv2D(
            in_channels,
            args.d_model,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias_attr=True,
            data_format="NHWC",
        )
        self.norm = nn.LayerNorm(args.d_model)  # 需要将channel放到最后一个维度上

    def forward(self, x):
        B, T, N, D = x.shape
        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        x_TAt = paddle.matmul(temporal_At, x.reshape([B, T, -1])).reshape([B, T, N, D])
        # SAt
        spatial_At = self.SAt(x_TAt)  # [B,N,N]
        # cheb gcn
        spatial_gcn = self.chebyAtt(x, spatial_At)  # [B,T,N,F]
        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn)  # [B,T,N,F]
        # residual shortcut
        x_residual = self.residual_conv(x)
        x_residual = self.norm(F.relu(x_residual + time_conv_output))  # [B,T,N,F]
        return x_residual
