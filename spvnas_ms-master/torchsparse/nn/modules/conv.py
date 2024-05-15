import math
from typing import Tuple, Union

import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer, Uniform

from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple

__all__ = ['Conv3d']


class Conv3d(nn.Cell):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transposed: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.transposed = transposed

        self.kernel_volume = int(np.prod(self.kernel_size))
        if self.kernel_volume > 1:
            self.kernel = Parameter(
                ops.Zeros()((self.kernel_volume, in_channels, out_channels), mindspore.float32))
        else:
            self.kernel = Parameter(ops.Zeros()((in_channels, out_channels), mindspore.float32))
        if bias:
            self.bias = Parameter(ops.Zeros()(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_ch annels}, kernel_size={kernel_size}'
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        if self.transposed:
            s += ', transposed=True'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume)
        # self.kernel = initializer(Uniform(std),
        #                           [self.kernel_volume, self.in_channels, self.out_channels],
        #                           mindspore.float32)
        self.kernel = initializer(Uniform(std),
                                  self.kernel.shape,
                                  mindspore.float32)
        # kernel_np = np.load('./torchsparse/nn/modules/kernel_torch.npy')
        # self.kernel = mindspore.Tensor(Parameter(kernel_np), dtype=mindspore.float32)
        # print(f"initialize.conv3d.weight: {self.kernel}")
        # print(f"initialize.conv3d.weight.shape: {self.kernel.shape}")
        if self.bias is not None:
            self.bias = initializer(Uniform(std),
                                  [self.out_channels],
                                  mindspore.float32)

    def construct(self, input: SparseTensor) -> SparseTensor:
        return F.conv3d(input,
                        self.kernel,
                        kernel_size=self.kernel_size,
                        bias=self.bias,
                        stride=self.stride,
                        dilation=self.dilation,
                        transposed=self.transposed)
