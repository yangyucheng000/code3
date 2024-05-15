import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
logits = Tensor(np.random.random((87245, 19)), ms.float32)
labels = Tensor(np.ones(87245).astype(np.int32))
print(f"logits: {logits.shape}, logits: {logits.dtype}")
print(f"labels_np: {labels.shape}, labels_np: {labels.dtype}")

valid_index_trans = (labels != 255).astype(ms.int32)
valid_index = valid_index_trans.nonzero().flatten()


output = loss(logits[valid_index], labels[valid_index])
print(output)