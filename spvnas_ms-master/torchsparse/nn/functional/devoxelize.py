import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore import ops
from mindspore import Tensor
from torchsparse.nn.cuda.devoxelize import SPDevoxelizeForward, SPDevoxelizeBackward

__all__ = ['spdevoxelize', 'calc_ti_weights']


def calc_ti_weights(coords: ms.Tensor,
                    idx_query: ms.Tensor,
                    scale: float = 1) -> ms.Tensor:
    p = coords
    F.stop_gradient(p)
    if scale != 1:
        pf = ops.floor(coords / scale) * scale
    else:
        pf = ops.floor(coords)
    pc = pf + scale

    x = p[:, 0].view(-1, 1)
    y = p[:, 1].view(-1, 1)
    z = p[:, 2].view(-1, 1)

    xf = pf[:, 0].view(-1, 1).astype(ms.float32)
    yf = pf[:, 1].view(-1, 1).astype(ms.float32)
    zf = pf[:, 2].view(-1, 1).astype(ms.float32)

    xc = pc[:, 0].view(-1, 1).astype(ms.float32)
    yc = pc[:, 1].view(-1, 1).astype(ms.float32)
    zc = pc[:, 2].view(-1, 1).astype(ms.float32)

    w0 = (xc - x) * (yc - y) * (zc - z)
    w1 = (xc - x) * (yc - y) * (z - zf)
    w2 = (xc - x) * (y - yf) * (zc - z)
    w3 = (xc - x) * (y - yf) * (z - zf)
    w4 = (x - xf) * (yc - y) * (zc - z)
    w5 = (x - xf) * (yc - y) * (z - zf)
    w6 = (x - xf) * (y - yf) * (zc - z)
    w7 = (x - xf) * (y - yf) * (z - zf)

    w = ops.concat([w0, w1, w2, w3, w4, w5, w6, w7], axis=1)
    w = w.transpose(1, 0)
    if scale != 1:
        w /= scale ** 3
    w[idx_query == -1] = 0
    w /= w.sum(axis=0) + 1e-8
    F.stop_gradient(w)

    return w


class DevoxelizeFunction(nn.Cell):
    def __init__(self):
        super(DevoxelizeFunction, self).__init__()
        self.sp_devoxelize_forward = SPDevoxelizeForward()
        self.sp_devoxelize_backward = SPDevoxelizeBackward()

    def construct(self, feats: Tensor, coords: Tensor,
                weights: Tensor) -> Tensor:

        if ms.get_context("device_target") == 'GPU':
            output = self.sp_devoxelize_forward(
                feats, coords.astype(ms.int32), weights)
        else:
            raise NotImplementedError

        return output


    # def bprop(self, feats: Tensor, coords: Tensor,
    #             weights: Tensor, output: Tensor, grad_output: Tensor):
    #
    #     if ms.get_context("device_target") == 'GPU':
    #         grad_feats = self.sp_devoxelize_backward(
    #             grad_output, coords.astype(ms.int32), weights, feats.shape[0])
    #     else:
    #         raise NotImplementedError
    #     return grad_feats, None, None


def spdevoxelize(feats: Tensor, coords: Tensor,
                 weights: Tensor) -> Tensor:
    return DevoxelizeFunction()(feats, coords, weights)
