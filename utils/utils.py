import copy
import math

import paddle.nn as nn
import paddle.optimizer as optimizer


def clones(module, N):
    """
    Produce N identical layers.
    :param module: nn.Layer
    :param N: int
    :return: paddle.nn.LayerList
    """
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class CosineAnnealingWithWarmupDecay(optimizer.lr.LRScheduler):
    def __init__(
        self, max_lr, min_lr, warmup_step, decay_step, last_epoch=-1, verbose=False
    ):
        self.decay_step = decay_step
        self.warmup_step = warmup_step
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(CosineAnnealingWithWarmupDecay, self).__init__(
            max_lr, last_epoch, verbose
        )

    def get_lr(self):
        if self.warmup_step > 0 and self.last_epoch <= self.warmup_step:
            return float(self.max_lr) * (self.last_epoch) / self.warmup_step

        if self.last_epoch > self.decay_step:
            return self.min_lr

        num_step_ = self.last_epoch - self.warmup_step
        decay_step_ = self.decay_step - self.warmup_step
        decay_ratio = float(num_step_) / float(decay_step_)
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class LinearAnnealingWithWarmupDecay(optimizer.lr.LRScheduler):
    def __init__(
        self, max_lr, min_lr, warmup_step, decay_step, last_epoch=-1, verbose=False
    ):
        self.decay_step = decay_step
        self.warmup_step = warmup_step
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(LinearAnnealingWithWarmupDecay, self).__init__(
            max_lr, last_epoch, verbose
        )

    def get_lr(self):
        if self.warmup_step > 0 and self.last_epoch <= self.warmup_step:
            return float(self.max_lr) * (self.last_epoch) / self.warmup_step

        if self.last_epoch > self.decay_step:
            return self.min_lr

        num_step_ = self.last_epoch - self.warmup_step
        decay_step_ = self.decay_step - self.warmup_step
        decay_ratio = float(num_step_) / float(decay_step_)
        coeff = 1.0 - decay_ratio
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
