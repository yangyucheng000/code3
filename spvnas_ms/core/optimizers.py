import mindspore as ms
import mindspore.nn as nn


# class SGDWithCosineWarmUp(nn.Optimizer):
#
#     def __init__(self, params, learning_rate=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False,
#                  loss_scale=1.0):
#         self.sgd = nn.SGD(params=params, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
#                           nesterov=nesterov, loss_scale=loss_scale)
#         self.dy_lr =