from typing import Any, List
from mindspore import Tensor
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
import numpy as np


from torchsparse import SparseTensor

__all__ = ['sparse_collate', 'sparse_collate_fn']


def sparse_collate(inputs: List[SparseTensor]) -> SparseTensor:
    coords, feats = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        if isinstance(x.coords, np.ndarray):
            x.coords = Tensor(x.coords)
        if isinstance(x.feats, np.ndarray):
            x.feats = Tensor(x.feats)

        assert isinstance(x.coords, Tensor), type(x.coords)
        assert isinstance(x.feats, Tensor), type(x.feats)
        assert x.stride == stride, (x.stride, stride)

        input_size = x.coords.shape[0]
        batch = mnp.full((input_size, 1),
                           k,
                           dtype=ms.int32)

        coords.append(ops.Concat(axis=1)((x.coords, batch)))
        feats.append(x.feats)

    coords = ops.Concat(axis=0)(coords)
    feats = ops.Concat(axis=0)(feats)
    output = SparseTensor(coords=coords, feats=feats, stride=stride)
    return output


def sparse_collate_fn(inputs: List[Any]) -> Any:
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            print('name:', name)
            print('type:', type(inputs[0][name]))
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn(
                    [input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = ops.Stack(axis=0)(
                    [Tensor(input[name]) for input in inputs])
            elif isinstance(inputs[0][name], Tensor):
                output[name] = ops.Stack(axis=0)([input[name] for input in inputs])
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs
