import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from torchsparse.nn.cuda.voxelize import SPVoxelize

__all__ = ['spvoxelize']


class VoxelizeFunction(nn.Cell):
    def __init__(self):
        super(VoxelizeFunction, self).__init__()
        self.sp_voxelize = SPVoxelize()

    def construct(self,
                  feats: Tensor,
                  coords: Tensor,
                  counts: Tensor) -> Tensor:

        output = self.sp_voxelize(feats, coords, counts)

        return output


def spvoxelize(feats: Tensor, coords: Tensor,
               counts: Tensor) -> Tensor:
    return VoxelizeFunction()(feats, coords, counts)
