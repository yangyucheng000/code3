from typing import Optional, Tuple, Union
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from torchsparse.nn.cuda.hash import SPHash

__all__ = ['sphash']


def sphash(coords: Tensor,
           offsets: Optional[Tensor] = None) -> Tensor:
    assert coords.dtype == ms.int32, coords.dtype
    assert coords.ndim == 2 and coords.shape[1] == 4, coords.shape

    if offsets is not None:
        assert offsets.dtype == ms.int32, offsets.dtype
        assert offsets.ndim == 2 and offsets.shape[1] == 3, offsets.shape

    return SPHash()(coords, offsets)
