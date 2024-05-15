from mindspore import nn
from mindspore.ops import relu

class FeatAvgPool(nn.Cell):
    """
    A class representing a feature average pooling layer.

    Args:
        input_size (int): The number of input channels.
        hidden_size (int): The number of output channels.
        kernel_size (int): The size of the pooling kernel.
        stride (int): The stride of the pooling operation.

    Attributes:
        conv (nn.Conv1d): The convolutional layer.
        pool (nn.AvgPool1d): The average pooling layer.

    """

    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1, stride=1,has_bias=True,pad_mode='pad')
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def construct(self, x):
        """
        Perform forward pass through the FeatAvgPool layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying convolution and average pooling.

        """
        x = x.swapaxes(1, 2)  # B, C, T
        return self.pool(relu(self.conv(x)))

def build_featpool(cfg):
    """
    Build the feature pooling layer based on the configuration.

    Args:
        cfg (CfgNode): The configuration node.

    Returns:
        FeatAvgPool: The feature pooling layer.
    """
    input_size = cfg.MODEL.TRM.FEATPOOL.INPUT_SIZE
    hidden_size = cfg.MODEL.TRM.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TRM.FEATPOOL.KERNEL_SIZE  # 4 for anet, 2 for tacos, 16 for charades
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.TRM.NUM_CLIPS
    return FeatAvgPool(input_size, hidden_size, kernel_size, stride)

if __name__ == '__main__':
    import numpy as np
    from mindspore import Tensor
    from mindspore import context
    # from ..config import cfg
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    featpool = FeatAvgPool(4096, 512, 2, 2)
    input = Tensor(np.random.rand(1, 512, 4096).astype(np.float32))
    output = featpool(input)
    print(output.shape)