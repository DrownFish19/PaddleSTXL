import paddle.nn as nn

from utils import clones


class EncoderLayer(nn.Layer):
    def __init__(self, size, self_attn, gcn):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.size = size

    def forward(self, x):
        x = x + self.norm1(self.self_attn(x, x, x))
        x = x + self.norm2(self.feed_forward_gcn(x))
        return x


class DecoderLayer(nn.Layer):
    def __init__(self, size, self_attn, cross_attn, gcn):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward_gcn = gcn
        self.norm = clones(nn.LayerNorm(size), 3)

    def forward(self, x, m):
        x = x + self.norm[0](self.self_attn(x, x, x, is_mask=True))
        x = x + self.norm[1](self.cross_attn(x, m, m))
        x = x + self.norm[2](self.feed_forward_gcn(x))
        return x


class Encoder(nn.Layer):
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Decoder(nn.Layer):
    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)
