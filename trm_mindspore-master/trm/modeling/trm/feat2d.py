import mindspore
from mindspore import nn
from mindspore import dtype as mstype
from mindspore import ops


class SparseMaxPool(nn.Cell):
    """
    SparseMaxPool is a module that performs sparse max pooling on the input tensor.
    """

    def __init__(self, pooling_counts, N):
        """
        Initialize the SparseMaxPool class.

        Args:
            pooling_counts (list): A list of integers representing the number of pooling operations for each level.
            N (int): The size of the input feature map.

        Returns:
            None
        """
        super().__init__()
        mask2d = ops.zeros((N, N), mstype.bool_)
        mask2d[list(range(N)), list(range(N))] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = list(range(0, N - offset, stride)), list(range(offset, N, stride))
                mask2d[i,j]=1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2, 1,pad_mode='pad') for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2,pad_mode='pad')] + [nn.MaxPool1d(2, 1,pad_mode='pad') for _ in range(c - 1)]
            )

        self.mask2d = mask2d #TODO: fix .to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def construct(self, x):
        """
        Forward pass of the 2D feature module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, D, N), where B is the batch size,
                              D is the number of features, and N is the number of elements.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, D, N, N), where B is the batch size,
                          D is the number of features, and N is the number of elements.
        """
        B, D, N = x.shape
        map2d = x.new_zeros((B, D, N, N))

        map2d[..., list(range(N)), list(range(N))] = x  # fill a diagonal line
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d


class SparseConv(nn.Cell):
    def __init__(self, pooling_counts, N, hidden_size):
        super().__init__()
        mask2d = ops.zeros((N, N), mstype.bool_)
        mask2d[list(range(N)), list(range(N))] = 1
        self.hidden_size = hidden_size
        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = list(range(0, N - offset, stride)), list(range(offset, N, stride))
                mask2d[i,j]=1
                maskij.append((i, j))
            stride *= 2

        self.convs = nn.CellList()
        self.convs.extend([nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=2, stride=1,has_bias=True,pad_mode='pad') for _ in range(pooling_counts[0])])
        for c in pooling_counts[1:]:
            self.convs.extend(
                [nn.Conv1d(hidden_size, hidden_size, 3, 2,has_bias=True,pad_mode='pad')] + [nn.Conv1d(hidden_size, hidden_size, 2, 1,has_bias=True,pad_mode='pad') for _ in range(c - 1)]
            )

        self.mask2d = mask2d #TODO: fix .to("cuda")
        self.maskij = maskij

    def construct(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros((B, D, N, N))
        map2d[..., list(range(N)), list(range(N))] = x  # fill a diagonal line
        for conv, (i, j) in zip(self.convs, self.maskij):
            x = conv(x)
            map2d[:, :, i, j] = x
        return map2d


def build_feat2d(cfg):
    """
    Build the 2D feature extraction module based on the provided configuration.

    Args:
        cfg (CfgNode): The configuration object.

    Returns:
        nn.Module: The 2D feature extraction module.

    Raises:
        NotImplementedError: If the specified feature 2D method is not implemented.
    """
    pooling_counts = cfg.MODEL.TRM.FEAT2D.POOLING_COUNTS  # [15,8,8] anet, [15] charades
    num_clips = cfg.MODEL.TRM.NUM_CLIPS  # 64 anet, 16 charades
    hidden_size = cfg.MODEL.TRM.FEATPOOL.HIDDEN_SIZE  # 512
    if cfg.MODEL.TRM.FEAT2D.NAME == "conv":
        return SparseConv(pooling_counts, num_clips, hidden_size)
    elif cfg.MODEL.TRM.FEAT2D.NAME == "pool":
        return SparseMaxPool(pooling_counts, num_clips)
    else:
        raise NotImplementedError("No such feature 2d method as %s" % cfg.MODEL.TRM.FEAT2D.NAME)

if __name__ == '__main__':
    import numpy as np
    from mindspore import Tensor
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    
    sparse_conv = SparseConv([15, 8, 8], 64, 512)
    sparse_max_pool = SparseMaxPool([15, 8, 8], 64)
    x = Tensor(np.random.rand(2, 512, 64), mindspore.float32)
    output1 = sparse_conv(x)
    output2 = sparse_max_pool(x)
    print(output1.shape)
    print(output2.shape)