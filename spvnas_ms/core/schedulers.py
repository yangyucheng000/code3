import numpy as np
# import torchpack.distributed as dist
from core.utils.config import configs
from mindspore.communication import get_group_size

__all__ = ['cosine_schedule_with_warmup']


# def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
#     batch_size *= dist.size()
#
#     if dist.size() == 1:
#         warmup_iters = 0
#     else:
#         warmup_iters = 1000 // dist.size()
#
#     if k < warmup_iters:
#         return (k + 1) / warmup_iters
#     else:
#         iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
#         ratio = (k - warmup_iters) / (num_epochs * iter_per_epoch)
#         return 0.5 * (1 + np.cos(np.pi * ratio))


def cosine_schedule_with_warmup(base_lr):
    num_epochs = configs.num_epochs
    batch_size = configs.batch_size
    dataset_size = configs.data.training_size

    batch_size *= get_group_size()
    if get_group_size() == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // get_group_size()

    iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
    total_iters = num_epochs * iter_per_epoch
    lr = []
    for k in range(total_iters):
        if k < warmup_iters:
            lr.append(base_lr * (k + 1) / warmup_iters)
        else:
            ratio = (k - warmup_iters) / total_iters
            lr.append(base_lr * 0.5 * (1 + np.cos(np.pi * ratio)))
    return lr
