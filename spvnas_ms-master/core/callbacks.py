# from typing import Any, Dict
#
# import numpy as np
# import torch
# from torchpack import distributed as dist
# from torchpack.callbacks.callback import Callback
#
# __all__ = ['MeanIoU']
#
#
# class MeanIoU(Callback):
#
#     def __init__(self,
#                  num_classes: int,
#                  ignore_label: int,
#                  *,
#                  output_tensor: str = 'outputs',
#                  target_tensor: str = 'targets',
#                  name: str = 'iou') -> None:
#         self.num_classes = num_classes
#         self.ignore_label = ignore_label
#         self.name = name
#         self.output_tensor = output_tensor
#         self.target_tensor = target_tensor
#
#     def _before_epoch(self) -> None:
#         self.total_seen = np.zeros(self.num_classes)
#         self.total_correct = np.zeros(self.num_classes)
#         self.total_positive = np.zeros(self.num_classes)
#
#     def _after_step(self, output_dict: Dict[str, Any]) -> None:
#         outputs = output_dict[self.output_tensor]
#         targets = output_dict[self.target_tensor]
#         outputs = outputs[targets != self.ignore_label]
#         targets = targets[targets != self.ignore_label]
#
#         for i in range(self.num_classes):
#             self.total_seen[i] += torch.sum(targets == i).item()
#             self.total_correct[i] += torch.sum((targets == i)
#                                                & (outputs == targets)).item()
#             self.total_positive[i] += torch.sum(outputs == i).item()
#
#     def _after_epoch(self) -> None:
#         for i in range(self.num_classes):
#             self.total_seen[i] = dist.allreduce(self.total_seen[i],
#                                                 reduction='sum')
#             self.total_correct[i] = dist.allreduce(self.total_correct[i],
#                                                    reduction='sum')
#             self.total_positive[i] = dist.allreduce(self.total_positive[i],
#                                                     reduction='sum')
#
#         ious = []
#
#         for i in range(self.num_classes):
#             if self.total_seen[i] == 0:
#                 ious.append(1)
#             else:
#                 cur_iou = self.total_correct[i] / (self.total_seen[i]
#                                                    + self.total_positive[i]
#                                                    - self.total_correct[i])
#                 ious.append(cur_iou)
#
#         miou = np.mean(ious)
#         if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
#             self.trainer.summary.add_scalar(self.name, miou * 100)
#         else:
#             print(ious)
#             print(miou)

import os
import time
from mindspore import save_checkpoint, context
from mindspore.train.callback import Callback
from core.utils.local_adapter import get_rank_id

class CallbackSaveByIoU(Callback):
    """SaveCallback"""
    def __init__(self, eval_model, ds_eval, eval_period=1, eval_start=1, save_path=None):
        """init"""
        super(CallbackSaveByIoU, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.mIoU = 0.
        self.mIoU_img = 0.
        self.eval_period = eval_period
        self.save_path = save_path
        self.eval_start = eval_start

    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        rank_id = get_rank_id()
        if ((cur_epoch + 1) % self.eval_period) == 0:
            if cur_epoch < self.eval_start:
                return
            print("Start evaluate...")
            result = self.model.eval(self.ds_eval, dataset_sink_mode=False)
            mIoU = result["mIoU"]
            if mIoU > self.mIoU:
                self.mIoU = mIoU
                file_name = f"best_model_dev_{rank_id}.ckpt"
                save_path = os.path.join(self.save_path, file_name)
                print("Save model...")
                save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=save_path)
            print(f"Device:{rank_id}, Epoch:{cur_epoch}, mIoU:{mIoU:.5f}")

    def end(self, run_context):
        _ = run_context.original_args()
        rank_id = get_rank_id()
        print(f"Device:{rank_id}, Best mIoU:{(self.mIoU*100):.2f}%")


class RecorderCallback(Callback):
    """Callback base class"""
    def __init__(self, recorder):
        self.recorder = recorder

    def step_begin(self, run_context):
        """Called before each step beginning."""
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        cur_time = time.time()
        cur_step = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)
        if context.get_context('device_target') == 'Ascend':
            log_str = ">>> {} Epoch[{:03d}/{:03d}] Step[{:04d}/{:04d}] Loss[{:.3f}] Time[{:.3f}s] SysTime[{}] ".format(
                "Train", int(cb_params.cur_epoch_num), int(cb_params.epoch_num),
                cur_step,
                int(cb_params.batch_num), float(cb_params.net_outputs[0]), cur_time - cb_params.init_time,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        else:
            log_str = ">>> {} Epoch[{:03d}/{:03d}] Step[{:04d}/{:04d}] Loss[{:.3f}] Time[{:.3f}s] SysTime[{}] ".format(
                "Train", int(cb_params.cur_epoch_num), int(cb_params.epoch_num),
                cur_step,
                int(cb_params.batch_num), float(cb_params.net_outputs), cur_time - cb_params.init_time,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if cur_step % 100 == 0:
            if context.get_context('device_target') == 'GPU':
                if self.recorder is not None:
                    self.recorder.logger.info(log_str)
            else:
                print(log_str)

    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params.epoch_init_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_time = time.time()
        if context.get_context('device_target') == 'Ascend':
            log_str = ">>> {} Epoch[{:03d}/{:03d}] Loss[{:.3f}] Time[{}s]".format(
                "Train", int(cb_params.cur_epoch_num), int(cb_params.epoch_num),
                float(cb_params.net_outputs[0]),
                time.strftime('%H:%M:%S', time.gmtime(cur_time - cb_params.epoch_init_time)))
            print(log_str)
        else:
            log_str = ">>> {} Epoch[{:03d}/{:03d}] Loss[{:.3f}] Time[{}s]".format(
                "Train", int(cb_params.cur_epoch_num), int(cb_params.epoch_num),
                float(cb_params.net_outputs),
                time.strftime('%H:%M:%S', time.gmtime(cur_time - cb_params.epoch_init_time)))
            if self.recorder is not None:
                self.recorder.logger.info(log_str)
