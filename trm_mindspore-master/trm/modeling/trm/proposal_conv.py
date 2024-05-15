import mindspore
from mindspore import nn,ops
from mindspore import dtype as mstype
import copy
def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d: B, D, N, N
    weight = ops.conv2d(mask2d[None, None, :, :].float(),
                          mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight


def get_padded_mask_and_weight(mask, conv):
    masked_weight = ops.round(ops.conv2d(copy.deepcopy(mask).float(), ops.ones((1, 1)+conv.kernel_size), stride=conv.stride, padding=conv.padding, dilation=conv.dilation,pad_mode='pad'))
    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]  #conv.kernel_size[0] * conv.kernel_size[1]
    padded_mask = masked_weight > 0
    return padded_mask, masked_weight

class ProposalConv(nn.Cell):
    def __init__(self, input_size, hidden_size, k, num_stack_layers, output_size, mask2d, dataset=None):
        super().__init__()
        self.num_stack_layers = num_stack_layers
        self.dataset = dataset
        self.mask2d = mask2d[None, None,:,:]
        # Padding to ensure the dimension of the output map2d
        first_padding = (k - 1) * num_stack_layers // 2
        self.bn = nn.CellList([nn.BatchNorm2d(hidden_size)])
        self.convs = nn.CellList(
            [nn.Conv2d(input_size, hidden_size, k, padding=first_padding,has_bias=True,pad_mode='pad')]
        )
        for _ in range(num_stack_layers - 1):
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k,has_bias=True,pad_mode='pad'))
            self.bn.append(nn.BatchNorm2d(hidden_size))
        self.conv1x1_iou = nn.Conv2d(hidden_size, output_size, 1,has_bias=True,pad_mode='pad')
        self.conv1x1_contrastive = nn.Conv2d(hidden_size, output_size, 1,has_bias=True,pad_mode='pad')

    def construct(self, x):
        # print('x:',x.shape)
        padded_mask = self.mask2d
        for i in range(self.num_stack_layers):
            x = ops.relu(self.bn[i](self.convs[i](x)))
            # print(f'{i} layer x:',x.shape)
            # print('padded_mask',padded_mask.shape)
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.convs[i])
            x = x * masked_weight
        out1 = self.conv1x1_contrastive(x)
        out2 = self.conv1x1_iou(x)
        return out1, out2


def build_proposal_conv(cfg, mask2d):
    input_size = cfg.MODEL.TRM.FEATPOOL.HIDDEN_SIZE
    hidden_size = cfg.MODEL.TRM.PREDICTOR.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TRM.PREDICTOR.KERNEL_SIZE
    num_stack_layers = cfg.MODEL.TRM.PREDICTOR.NUM_STACK_LAYERS
    output_size = cfg.MODEL.TRM.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME
    return ProposalConv(input_size, hidden_size, kernel_size, num_stack_layers, output_size, mask2d, dataset_name)


if __name__ =='__main__':
    N = 256
    import numpy as np
    from mindspore import Tensor
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    
    mask2d = ops.zeros((N, N), mstype.bool_)
    mask2d[list(range(N)), list(range(N))] = 1
    conv = ProposalConv(256, 256, 3, 3, 256, mask2d)
    
    x = Tensor(np.random.rand(1, 256, 256, 256).astype(np.float32))
    out1, out2 = conv(x)
    print(out1.shape, out2.shape)