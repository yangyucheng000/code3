from typing import Callable
from core.utils.config import configs
import mindspore as ms
import mindspore.nn as nn
from core.schedulers import cosine_schedule_with_warmup

# import torch
# import torch.optim
# from torch import nn
# from torchpack.utils.config import configs
# from torchpack.utils.typing import Dataset, Optimizer, Scheduler

# __all__ = [
#     'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
#     'make_scheduler'
# ]
__all__ = [
    'make_dataset', 'make_criterion', 'make_optimizer'
]

def make_dataset():
    if configs.dataset.name == 'semantic_kitti':
        from core.datasets import SemanticKITTI
        dataset = SemanticKITTI(root=configs.dataset.root,
                                num_points=configs.dataset.num_points,
                                voxel_size=configs.dataset.voxel_size)
    elif configs.dataset.name == 'dummy_kitti':
        from core.datasets import DummyKITTI
        dataset = DummyKITTI(num_points=configs.dataset.num_points,
                             voxel_size=configs.dataset.voxel_size)
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


def make_model():
    if configs.model.name == 'minkunet':
        from core.models.semantic_kitti import MinkUNet
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = MinkUNet(num_classes=configs.data.num_classes, cr=cr)
    elif configs.model.name == 'spvcnn':
        from core.models.semantic_kitti import SPVCNN
        from core.models.semantic_kitti import SPVCNN_MS
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SPVCNN_MS(num_classes=configs.data.num_classes,
        # model = SPVCNN(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    else:
        raise NotImplementedError(configs.model.name)
    return model
#
#
def make_criterion():
    if configs.criterion.name == 'cross_entropy':
        from core.criterions import CrossEntropyLossWithIgnored
        criterion = CrossEntropyLossWithIgnored(
            sparse=True, reduction='mean', ignore_index=configs.criterion.ignore_index
        )
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion

def make_optimizer(model):
    if configs.optimizer.name == 'sgd':
        # dynamic_lr = cosine_schedule_with_warmup(configs.optimizer.lr)
        optimizer = nn.SGD(model.trainable_params(),
                           learning_rate=configs.optimizer.lr,
                           momentum=configs.optimizer.momentum,
                           weight_decay=configs.optimizer.weight_decay,
                           nesterov=configs.optimizer.nesterov)
    # elif configs.optimizer.name == 'adam':
    #     optimizer = torch.optim.Adam(
    #         model.parameters(),
    #         lr=configs.optimizer.lr,
    #         weight_decay=configs.optimizer.weight_decay)
    # elif configs.optimizer.name == 'adamw':
    #     optimizer = torch.optim.AdamW(
    #         model.parameters(),
    #         lr=configs.optimizer.lr,
    #         weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer

# def make_scheduler(optimizer: Optimizer) -> Scheduler:
#     if configs.scheduler.name == 'none':
#         scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
#                                                       lr_lambda=lambda epoch: 1)
#     elif configs.scheduler.name == 'cosine':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=configs.num_epochs)
#     elif configs.scheduler.name == 'cosine_warmup':
#         from functools import partial
#
#         from core.schedulers import cosine_schedule_with_warmup
#         scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer,
#             lr_lambda=partial(cosine_schedule_with_warmup,
#                               num_epochs=configs.num_epochs,
#                               batch_size=configs.batch_size,
#                               dataset_size=configs.data.training_size))
#     else:
#         raise NotImplementedError(configs.scheduler.name)
#     return scheduler
