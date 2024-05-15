from typing import Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple
import numpy as np

__all__ = ['spdownsample']


def spdownsample(
        coords: ms.Tensor,
        stride: Union[int, Tuple[int, ...]] = 2,
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        tensor_stride: Union[int, Tuple[int, ...]] = 1) -> ms.Tensor:
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    tensor_stride = make_ntuple(tensor_stride, ndim=3)

    sample_stride = [stride[k] * tensor_stride[k] for k in range(3)]
    sample_stride = ops.expand_dims(ms.Tensor(sample_stride, dtype=ms.int32), 0)

    if all(stride[k] in [1, kernel_size[k]] for k in range(3)):
        coords = ops.deepcopy(coords)
        coords[:, :3] = coords[:, :3] // sample_stride * sample_stride
    else:
        offsets = get_kernel_offsets(kernel_size,
                                     tensor_stride)
        kernel_volume = offsets.shape[0]

        # coords_min = ops.min(coords[:, :3], axis=0, keep_dims=True).values # TODO[lichen] .values in mindspore
        _, coords_min = ops.min(coords[:, :3], axis=0, keepdims=True)  # ms2

        x = ops.repeat_elements(ops.expand_dims(coords[:, :3], axis=1), kernel_volume, 1) + offsets
        b = coords[:, 3:].repeat(1, kernel_volume)
        coords = ops.concat([x.view(-1, 3), b.view(-1, 1)], axis=1)

        mask = (coords[:, :3] % sample_stride == 0)
        mask &= (coords[:, :3] >= coords_min)
        mask = mask.all(axis=1)
        coords = coords[mask]

    # This makes sure that the points will be ordered with respect to the batch
    # index, but this will not affect the correctness of the result.
    coords = coords[:, [3, 0, 1, 2]]
    # coords = torch.unique(coords, dim=0) # TODO[lichen]: unique with dim in mindspore
    # coords = ops.unique(coords)[0]  # stf: unique on dim=0, it is the same
    # coords, _ = ops.sort(coords, axis=0)
    # coords = ops.unique_consecutive(coords, axis=0)
    coords_ms = Tensor.from_numpy(np.unique(coords.asnumpy(), axis=0))
    coords = coords_ms[:, [1, 2, 3, 0]]
    return coords
