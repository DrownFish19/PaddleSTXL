import copy

import paddle.nn as nn


def clones(module, N):
    """
    Produce N identical layers.
    :param module: nn.Layer
    :param N: int
    :return: paddle.nn.LayerList
    """
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])
