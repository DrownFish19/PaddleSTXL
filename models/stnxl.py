import math

import paddle
import paddle.nn as nn


class STNXL(nn.Layer):
    def __init__(self, args, adj_matrix):
        super(STNXL, self).__init__()
        self.args = args
        self.adj_matrix = adj_matrix
        self.sc_matrix = sc_matrix
        self.lookup_index = lookup_index
        # 初始化其他参数和层...
        # ...

    def forward(self, x):
        # 前向传播的计算过程...
        # ...
        return output
