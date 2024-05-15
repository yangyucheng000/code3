import argparse
import os
import mindspore as ms
from mindspore import context
from trm.config import cfg
from trm.data import make_data_loader
from trm.engine.inference import inference
from trm.modeling import build_model
from loguru import logger
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
def main():
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # distributed = num_gpus > 1

    # if distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="env://"
    #     )
    #     synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    # logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    
    model = build_model(cfg)
    model.to_float(ms.float32)
    # model.to(cfg.MODEL.DEVICE)
    model.set_train(False)

    # for epoch in [10, 11, 9, 12, 8, 13, 7, 14, 6, 15, 5, 16, 4, 17, 3, 18, 2, 1]:
    output_dir = cfg.OUTPUT_DIR
    
    ms.load_checkpoint(args.ckpt, model)
    params = ms.load_checkpoint(args.ckpt)
    missing,unexpect = ms.load_param_into_net(model, params)
    logger.info(f'missing keys: {missing}')
    logger.info(f'unexpected keys: {unexpect}')
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    # ckpt.replace('pool_model_12e.pth', 'pool_model_%de.pth'%epoch)
    logger.info("load from %s"%ckpt)


    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False)
    _ = inference(
        cfg,
        model,
        data_loaders_val,
        dataset_name=dataset_names,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
    )

if __name__ == "__main__":
    main()
