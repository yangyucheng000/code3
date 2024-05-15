from typing import List
# import torch
import mindspore as ms
from torchsparse.tensor import SparseTensor

__all__ = ['cat']


def cat(inputs: List[SparseTensor]) -> SparseTensor:
    feats = ms.ops.concat([input.feats for input in inputs], axis=1)
    output = SparseTensor(coords=inputs[0].coords,
                          feats=feats,
                          stride=inputs[0].stride)
    output.cmaps = inputs[0].cmaps
    output.kmaps = inputs[0].kmaps
    return output
