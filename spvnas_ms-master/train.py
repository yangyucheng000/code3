import argparse
import random
import sys
import os

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context, set_seed, Model, ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from core.utils.config import configs
from core.utils.local_adapter import execute_distributed, distributed
from core.trainers import CustomWithLossCell

import numpy as np
# from torchpack import distributed as dist
# from torchpack.callbacks import InferenceRunner, MaxSaver, Saver
# from torchpack.environ import auto_set_run_dir, set_run_dir
# # from torchpack.utils.config import configs
# from torchpack.utils.logging import logger

from core import builder
# from core.callbacks import MeanIoU
# from core.trainers import SemanticKITTITrainer

order = "configs/semantic_kitti/spvcnn/cr0p5.yaml --distributed False"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()


    configs.load(args.config, recursive=True)
    configs.update(opts)
    configs.update(vars(args))

    # context.set_context(device_target='GPU')
    print("-->GPU数量: ", configs.n_gpus)
    rank = int(os.getenv('RANK_ID', '0'))
    if configs.n_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        init()
        rank = get_rank()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        # is_distributed = True
        execute_distributed()
        # if rank == 0:
        #     recorder = Recorder(settings, settings.save_path)
        # else:
        #     recorder = 0
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU",
                            device_id=int(configs.gpu[0]))
        # is_distributed = False
        # recorder = Recorder(settings, settings.save_path)
    cuda_path = os.path.join(os.path.dirname(__file__), "torchsparse/nn/cuda")
    os.environ["MS_CUSTOM_AOT_WHITE_LIST"] = cuda_path
    # if args.run_dir is None:
    #     args.run_dir = auto_set_run_dir()
    # else:
    #     set_run_dir(args.run_dir)

    # logger.info(' '.join([sys.executable] + sys.argv))
    # logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = np.random.randint(np.int32) % (2 ** 32 - 1)

    # seed = configs.train.seed + dist.rank(
    # ) * configs.workers_per_gpu * configs.num_epochs
    seed = configs.train.seed + rank * configs.workers_per_gpu * configs.num_epochs
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    # configs.dataset.name = 'dummy_kitti'
    # print('=====================using dummpy_kitti==================================')
    dataset = builder.make_dataset()
    dataflow = {}
    for split in dataset:
        if distributed:
            rank_size = get_group_size()
            sampler = ms.dataset.DistributedSampler(
                num_shards=rank_size,
                shard_id=rank,
                shuffle=(split == 'train'),
            )
            dataflow[split] = ds.GeneratorDataset(
                dataset[split],
                column_names=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                sampler=sampler
            )
            dataflow[split] = dataflow[split].batch(
                batch_size=configs.batch_size,
                num_parallel_workers=configs.workers_per_gpu,
                per_batch_map=dataset[split].per_batch_map,
                # output_columns=['lidar', 'targets', 'targets_mapped', 'inverse_map', 'file_name']
                output_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name', 'num_vox', 'num_pts']
            )
        else:
            dataflow[split] = ds.GeneratorDataset(
                dataset[split],
                column_names=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                shuffle=(split == 'train')
            )
            dataflow[split] = dataflow[split].batch(
                batch_size=configs.batch_size,
                num_parallel_workers=configs.workers_per_gpu,
                per_batch_map=dataset[split].per_batch_map,
                # input_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                output_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name', 'num_vox', 'num_pts']
                # output_columns=['feed_dict_list']
            )

    # from visualize import visualize_pcd
    # for fd_tuple in dataflow['train'].create_tuple_iterator(output_numpy=True):
    #     feed_dict = dataset['train'].collate_fn(*fd_tuple)
    #     lidar = feed_dict['lidar']
    #     num_vox = [80000, 80000]
    #     targets = ms.ops.cast(feed_dict['targets'].F, ms.int64)
    #     cur = 0
    #     for n in num_vox:
    #         pts = lidar.F[cur:cur+n, :3]
    #         label = targets[cur:cur+n]
    #         visualize_pcd(xyz=pts, target=label)
    #         cur += n

    net = builder.make_model()
    # from visualize import visualize_pcd
    # for fd_tuple in dataflow['train'].create_tuple_iterator(output_numpy=True):
    #     feed_dict = dataset['train'].collate_fn(*fd_tuple)
    #     lidar = feed_dict['lidar']
    #     num_vox = [80000, 80000]
    #     targets = ms.ops.cast(feed_dict['targets'].F, ms.int64)
    #     cur = 0
    #     for n in num_vox:
    #         pts = lidar.F[cur:cur+n, :3]
    #         label = targets[cur:cur+n]
    #         visualize_pcd(xyz=pts, target=label)
    #         cur += n

    # # if configs.distributed:
    # #     model = torch.nn.parallel.DistributedDataParallel(
    # #         model, device_ids=[dist.local_rank()], find_unused_parameters=True)
    #
    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(net)
    # scheduler = builder.make_scheduler(optimizer)
    loss_net = CustomWithLossCell(net, criterion)
    # loss_scale_manager = ms.DynamicLossScaleManager()
    # model = ms.Model(loss_net, amp_level='O2', loss_scale_manager=loss_scale_manager, optimizer=optimizer)
    model = ms.Model(loss_net, optimizer=optimizer)
    model.train(1, dataflow["train"], dataset_sink_mode=False)

    #
    # trainer = SemanticKITTITrainer(model=model,
    #                                criterion=criterion,
    #                                optimizer=optimizer,
    #                                scheduler=scheduler,
    #                                num_workers=configs.workers_per_gpu,
    #                                seed=seed,
    #                                amp_enabled=configs.amp_enabled)
    # trainer.train_with_defaults(
    #     dataflow['train'],
    #     num_epochs=configs.num_epochs,
    #     callbacks=[
    #         InferenceRunner(
    #             dataflow[split],
    #             callbacks=[
    #                 MeanIoU(name=f'iou/{split}',
    #                         num_classes=configs.data.num_classes,
    #                         ignore_label=configs.data.ignore_label)
    #             ],
    #         ) for split in ['test']
    #     ] + [
    #         MaxSaver('iou/test'),
    #         Saver(),
    #     ])


if __name__ == '__main__':
    main()
