import argparse
import os
import random
import sys
sys.path.append('./')
from trm.data import make_data_loader
from trm.config import cfg
from trm.engine.inference import inference
from trm.engine.trainer import do_train
from trm.modeling import build_model
from trm.utils.checkpoint import TrmCheckpointer
from trm.utils.miscellaneous import mkdir, save_config
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor

from mindspore import nn, context, Tensor
import mindspore
# mindspore.set_context(env_config_path="./mindspore_config.json")
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
from mindspore.communication import init



import random

import numpy

import mindspore

mindspore.set_seed(42)
numpy.random.seed(42)
random.seed(42)


from loguru import logger
def train(cfg):
    model = build_model(cfg)
    learning_rate = cfg.SOLVER.LR * 1.0
    
    data_loader = make_data_loader(cfg, is_train=True)


    bert_params = list(filter(lambda x: 'bert' in x.name, model.trainable_params()))
    base_params = list(filter(lambda x: 'bert' not in x.name, model.trainable_params()))

    # 基于多项式衰减函数计算学习率
    polynomial_decay_lr1 = nn.PolynomialDecayLR(learning_rate=learning_rate,      # 学习率初始值
                                            end_learning_rate=learning_rate*0.001, # 学习率最终值
                                            decay_steps=4,          # 衰减的step数
                                            power=0.5)              # 多项式幂
    
    polynomial_decay_lr2 = nn.PolynomialDecayLR(learning_rate=learning_rate*0.1,      # 学习率初始值
                                            end_learning_rate=learning_rate*0.0001, # 学习率最终值
                                            decay_steps=4,          # 衰减的step数
                                            power=0.5)              # 多项式幂
    
    group_params = [{'params': bert_params, 'weight_decay': 0.01, 'lr': polynomial_decay_lr2},
                {'params': base_params, 'lr': polynomial_decay_lr1}]
    
    optimizer = nn.AdamWeightDecay(group_params, learning_rate=learning_rate, weight_decay=1e-5)
    output_dir = cfg.OUTPUT_DIR
    # checkpointer = TrmCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
    arguments = {"epoch": 1}


    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_for_period=False)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        output_dir,
        checkpoint_period,
        test_period,
        arguments,
        group_params
    )
    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
        )


def main():
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    seed = 25285
    random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.distributed = num_gpus > 1

    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="env://"
    #     )
    #     synchronize()

    cfg.merge_from_file(args.config_file)
    print('config_file',args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        # logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg)

    if not args.skip_test:
        run_test(cfg, model)


if __name__ == "__main__":
    #mp.set_start_method('spawn')
    #
    main()
