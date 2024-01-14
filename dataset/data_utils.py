import math

import numpy as np
import paddle


class ScalerStd(object):
    """
    Desc: Normalization utilities with std mean
    """

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = (
            paddle.tensor(self.mean).type_as(data).to(data.device)
            if paddle.is_tensor(data)
            else self.mean
        )
        std = (
            paddle.tensor(self.std).type_as(data).to(data.device)
            if paddle.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """

        mean = paddle.tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


class ScalerMinMax(object):
    """
    Desc: Normalization utilities with min max
    """

    def __init__(self):
        self.min = 0.0
        self.max = 1.0

    def fit(self, data):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        _min = paddle.to_tensor(self.min) if paddle.is_tensor(data) else self.min
        _max = paddle.to_tensor(self.max) if paddle.is_tensor(data) else self.max
        data = 1.0 * (data - _min) / (_max - _min)
        return 2.0 * data - 1.0

    def inverse_transform(self, data, axis=None):
        # type: (paddle.tensor, None) -> paddle.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """

        _min = paddle.to_tensor(self.min) if paddle.is_tensor(data) else self.min
        _max = paddle.to_tensor(self.max) if paddle.is_tensor(data) else self.max
        data = (data + 1.0) / 2.0

        if axis is None:
            return 1.0 * data * (_max[axis] - _min[axis]) + _min[axis]
        else:
            return 1.0 * data * (_max[axis] - _min[axis]) + _min[axis]


# 简化的哈弗辛公式来计算地球上两点之间的距离
def haversine(lon1, lat1, lon2, lat2):
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # 哈弗辛公式
    dis_lon = lon2 - lon1
    dis_lat = lat2 - lat1
    a = (
        math.sin(dis_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dis_lon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r
