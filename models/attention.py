import math
from copy import deepcopy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class VanillaAttention(nn.Layer):
    def __init__(self):
        super(VanillaAttention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query:  [B,N,H,T,D]
        :param key: [B,N,H,T,D]
        :param value: [B,N,H,T,D]
        :param mask: [B,1,1,T2,T2]
        :param dropout:
        :return: [B,N,H,T1,d], [B,N,H,T1,T2]
        """
        B, N, H, T, D = query.shape
        # [B,N,H,T1,T2]
        scores = paddle.matmul(query, key, transpose_y=True) / math.sqrt(D)

        if mask is not None:
            scores = scores + mask
        p_attn = F.softmax(scores, axis=-1)  # [B,N,H,T1,T2]
        if dropout is not None:
            p_attn = dropout(p_attn)

        return paddle.matmul(p_attn, value)  # [B,N,H,T1,T2] * [B,N,H,T1,D]


class SmoothAttention(nn.Layer):
    def __init__(self, args):
        super(SmoothAttention, self).__init__()
        # [N, K]
        self.corr_values = paddle.create_parameter(
            [args.num_nodes, args.node_top_k], dtype=paddle.get_default_dtype()
        )
        self.corr_indices = paddle.create_parameter(
            [args.num_nodes, args.node_top_k],
            dtype=paddle.int64,
            is_bias=True,
        )

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query:  [B,N,H,T,D]
        :param key: [B,N,H,T,D]
        :param value: [B,N,H,T,D]
        :param mask: [B,1,1,T2,T2]
        :param dropout:
        :return: [B,N,H,T1,d], [B,N,H,T1,T2]
        """
        B, N, H, T, D = query.shape

        # [B, N ,H, T, D] => [k, B, N, H, T, D]
        key_top_k = key[:, self.corr_indices, :, :, :]
        key = paddle.einsum("nk, kbnhtd -> bnhtd", self.corr_values, key_top_k)

        # [B,N,H,T1,T2]
        scores = paddle.matmul(query, key, transpose_y=True) / math.sqrt(D)

        if mask is not None:
            scores = scores + mask
        p_attn = F.softmax(scores, axis=-1)  # [B,N,H,T1,T2]
        if dropout is not None:
            p_attn = dropout(p_attn)

        return paddle.matmul(p_attn, value)  # [B,N,H,T1,T2] * [B,N,H,T1,D]


class MultiHeadAttentionAwareTemporalContext(nn.Layer):
    def __init__(self, args, query_conv_type="1DConv", key_conv_type="1DConv"):
        """
        input shape: [B,T,N,D]
        """
        super(MultiHeadAttentionAwareTemporalContext, self).__init__()
        self.training_args = args
        assert args.d_model % args.head == 0
        self.head_dim = args.d_model // args.head
        self.heads = args.head

        # 构建aware_temporal_context
        self.padding_causal = args.kernel_size - 1
        self.padding_1DConv = (args.kernel_size - 1) // 2
        self.query_conv_type = query_conv_type
        self.key_conv_type = key_conv_type

        conv_1d = nn.Conv2D(
            args.d_model,
            args.d_model,
            (args.kernel_size, 1),
            padding=(self.padding_1DConv, 0),
            bias_attr=True,
            data_format="NHWC",
        )

        conv_causal = nn.Conv2D(
            args.d_model,
            args.d_model,
            (args.kernel_size, 1),
            padding=(self.padding_causal, 0),
            bias_attr=True,
            data_format="NHWC",
        )
        if query_conv_type == "1DConv":
            self.query_conv = deepcopy(conv_1d)
        else:
            self.query_conv = deepcopy(conv_causal)

        if key_conv_type == "1DConv":
            self.key_conv = deepcopy(conv_1d)
            self.value_conv = deepcopy(conv_1d)
        else:
            self.key_conv = deepcopy(conv_causal)
            self.value_conv = deepcopy(conv_causal)

        self.out_mlp = nn.Linear(args.d_model, args.d_model, bias_attr=True)

        self.dropout = nn.Dropout(p=args.dropout)

        self.attention = VanillaAttention()
        self.attention_sm = SmoothAttention(self.training_args)
        self.attention_type = args.attention

    def subsequent_mask(self, size):
        """
        mask out subsequent positions.
        :param size: int
        :return: (1, size, size)
        """
        mask = paddle.full(
            [1, size, size],
            paddle.finfo(paddle.float32).min,
            dtype=paddle.float32,
        )
        mask = paddle.triu(mask, diagonal=1)
        return mask

    def forward(self, query, key, value, is_mask=False):
        """
        Args:
            query: [B,T,N,D]
            key: [B,T,N,D]
            value: [B,T,N,D]
            is_mask: bool
            query_multi_segment: bool
            key_multi_segment: bool

        Returns: [B,T,N,D]

        """
        B, T, N, D = query.shape
        B, T2, N, D = key.shape
        if is_mask:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            # (1, T', T')
            mask = self.subsequent_mask(T).unsqueeze([0, 1])
        else:
            mask = None

        query = self.query_conv(query)  # B, T, N, D
        key = self.key_conv(key)  # B, T, N, D
        value = self.value_conv(value)  # B, T, N, D

        if self.query_conv_type == "causal":
            query = query[:, : -self.padding_causal, :, :]
        if self.key_conv_type == "causal":
            key = key[:, : -self.padding_causal, :, :]
            value = value[:, : -self.padding_causal, :, :]

        # convert [B,T,N,D] to [B,T,N,H,D] to [B,N,H,T,D]
        multi_head_shape = [B, -1, N, self.heads, self.head_dim]
        perm = [0, 2, 3, 1, 4]
        query = query.reshape(multi_head_shape).transpose(perm)
        key = key.reshape(multi_head_shape).transpose(perm)
        value = value.reshape(multi_head_shape).transpose(perm)

        # [B,N,H,T,d]
        # x = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x_sm = self.attention_sm(query, key, value, mask=mask, dropout=self.dropout)
        x = x_sm
        # x = x + x_sm

        # [B,N,T,D]
        x = x.transpose([0, 3, 1, 2, 4]).reshape([B, -1, N, self.heads * self.head_dim])
        return self.out_mlp(x)
