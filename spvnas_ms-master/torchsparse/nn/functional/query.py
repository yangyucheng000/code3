import mindspore as ms
import mindspore.ops as ops
from torchsparse.nn.cuda.others import SPHashQuery

def sphashquery(queries: ms.Tensor,
                references: ms.Tensor) -> ms.Tensor:

    sizes = queries.shape
    queries = queries.view(-1)

    # indices = ops.arange(start=0, stop=references.shape[0], rtype=ms.int64)  # ms1
    indices = ops.arange(start=0, end=references.shape[0], dtype=ms.int64)  # ms2
    output = SPHashQuery()(queries, references, indices)

    output = (output - 1).view(sizes)
    return output
