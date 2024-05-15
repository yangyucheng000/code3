import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class CrossEntropyLossWithIgnored(nn.Cell):

    def __init__(self, sparse=False, reduction='none', ignore_index=255):
        super(CrossEntropyLossWithIgnored, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction=reduction)

    def construct(self, logits, labels):
        valid_index_trans = (labels != self.ignore_index).astype(ms.int32)
        valid_index = valid_index_trans.nonzero().flatten()

        # print(f"loss.valid_index: {valid_index}")
        # print(f"loss.valid_index.shape: {valid_index.shape}, loss.valid_index.dtype: {valid_index.dtype}")

        ce = self.ce(logits[valid_index], labels[valid_index])
        print('=============ce: %f ================' % ce)
        return ce