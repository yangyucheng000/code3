from typing import Any, Callable, Dict
import numpy as np
import mindspore as ms
import mindspore.nn as nn

from core.datasets.semantic_kitti import SemanticKITTIInternal
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn

# import torch
# from torch import nn
# from torch.cuda import amp
# from torchpack.train import Trainer
# from torchpack.utils.typing import Optimizer, Scheduler
#
# __all__ = ['SemanticKITTITrainer']
#
#
# class SemanticKITTITrainer(Trainer):
#
#     def __init__(self,
#                  model: nn.Module,
#                  criterion: Callable,
#                  optimizer: Optimizer,
#                  scheduler: Scheduler,
#                  num_workers: int,
#                  seed: int,
#                  amp_enabled: bool = False) -> None:
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.num_workers = num_workers
#         self.seed = seed
#         self.amp_enabled = amp_enabled
#         self.scaler = amp.GradScaler(enabled=self.amp_enabled)
#         self.epoch_num = 1
#
#     def _before_epoch(self) -> None:
#         self.model.train()
#         self.dataflow.sampler.set_epoch(self.epoch_num - 1)
#
#         self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
#             self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)
#
#     def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
#         _inputs = {}
#         for key, value in feed_dict.items():
#             if 'name' not in key:
#                 _inputs[key] = value.cuda()
#
#         inputs = _inputs['lidar']
#         targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
#
#         with amp.autocast(enabled=self.amp_enabled):
#             outputs = self.model(inputs)
#
#             if outputs.requires_grad:
#                 loss = self.criterion(outputs, targets)
#
#         if outputs.requires_grad:
#             self.summary.add_scalar('loss', loss.item())
#
#             self.optimizer.zero_grad()
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#             self.scheduler.step()
#         else:
#             invs = feed_dict['inverse_map']
#             all_labels = feed_dict['targets_mapped']
#             _outputs = []
#             _targets = []
#             for idx in range(invs.C[:, -1].max() + 1):
#                 cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
#                 cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
#                 cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
#                 outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
#                 targets_mapped = all_labels.F[cur_label]
#                 _outputs.append(outputs_mapped)
#                 _targets.append(targets_mapped)
#             outputs = torch.cat(_outputs, 0)
#             targets = torch.cat(_targets, 0)
#
#         return {'outputs': outputs, 'targets': targets}
#
#     def _after_epoch(self) -> None:
#         self.model.eval()
#
#     def _state_dict(self) -> Dict[str, Any]:
#         state_dict = {}
#         state_dict['model'] = self.model.state_dict()
#         state_dict['scaler'] = self.scaler.state_dict()
#         state_dict['optimizer'] = self.optimizer.state_dict()
#         state_dict['scheduler'] = self.scheduler.state_dict()
#         return state_dict
#
#     def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#         self.model.load_state_dict(state_dict['model'])
#         self.scaler.load_state_dict(state_dict.pop('scaler'))
#         self.optimizer.load_state_dict(state_dict['optimizer'])
#         self.scheduler.load_state_dict(state_dict['scheduler'])
#
#     def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
#         pass

class CustomWithLossCell(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, pc, feat, labels, pc_, labels_, inverse_map, file_name, num_vox, num_pts):

        feed_dict = SemanticKITTIInternal.collate_fn(pc, feat, labels, pc_, labels_, inverse_map, file_name, num_vox, num_pts)
        input = feed_dict['lidar']
        targets = feed_dict['targets'].F.astype(ms.int64)

        print(f"net_loss.backbone.input: {input.F}")
        print(f"net_loss.backbone.input.shape: {input.F.shape}, net_loss.backbone.input.dtype: {input.F.dtype}")

        output = self._backbone(input)  # 前向计算得到网络输出

        # print(f"net_loss.output: {output}")
        # print(f"net_loss.output.shape: {output.shape}, net_loss.output.dtype: {output.dtype}")
        # print(f"net_loss.label.F.shape: {targets.shape}, net_loss.label.F.dtype: {targets.dtype}")

        return self._loss_fn(output, targets)  # 得到多损失值
