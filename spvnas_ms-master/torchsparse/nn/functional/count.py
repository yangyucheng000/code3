import mindspore as ms
import mindspore.ops as ops
from torchsparse.nn.cuda.others import SPCount

__all__ = ['spcount']

def spcount(coords: ms.Tensor, num: ms.Tensor) -> ms.Tensor:
    return SPCount()(coords.astype("int32"), ops.Zeros()((num), ms.int32))
