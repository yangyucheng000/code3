"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import mindspore as ms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.ops.boxes import box_iou
sys.path.append('detr')
from upt_tip_cache_model_free_finetune_distill3 import build_detector

from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory
import pdb
from hico_text_label import hico_unseen_index
import vcoco_text_label, hico_text_label

warnings.filterwarnings("ignore")

def tranverse_and_get_hoi_cooccurence(dataset):
    category = dataset.num_interation_cls
    hoi_cooccurence = torch.zeros(category, category)
    for anno in dataset._anno:
        num_gt = len(anno['hoi'])
        for i in range(num_gt):
            for j in range(i+1, num_gt):
                ## need to judge if anno['hoi'][i] and anno['hoi'][j] are the same pair
                h_iou = box_iou(torch.as_tensor(anno['boxes_h'][i:i+1]), torch.as_tensor(anno['boxes_h'][j:j+1]))
                o_iou = box_iou(torch.as_tensor(anno['boxes_o'][i:i+1]), torch.as_tensor(anno['boxes_o'][j:j+1]))
                if min(h_iou.item(), o_iou.item()) > 0.5:
                    if anno['hoi'][i] == anno['hoi'][j]:
                        continue
                    hoi_cooccurence[anno['hoi'][i],anno['hoi'][j]] += 1
                    hoi_cooccurence[anno['hoi'][j],anno['hoi'][i]] += 1
    hoi_cooccurence = hoi_cooccurence.t() / (hoi_cooccurence.sum(dim=-1) + 1e-9)
    hoi_cooccurence = hoi_cooccurence.t()   
    return hoi_cooccurence

def hico_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
    class_corr = []
    for i, (k, v) in enumerate(hico_text_label.hico_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr

def vcoco_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
    class_corr = []
    for i, (k, v) in enumerate(vcoco_text_label.vcoco_hoi_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr

def vcoco_object_n_verb_to_interaction(num_object_cls, num_action_cls, class_corr):
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([num_object_cls, num_action_cls], None)
        for i, j, k in class_corr:
            lut[j, k] = i
        return lut.tolist()

def swig_object_n_verb_to_interaction(num_object_cls, num_action_cls, class_corr):
        """
        The interaction classes corresponding to an object-verb pair

        class_corr: List[(hoi_id, object_id, action_id)]

        Returns:
            list[list[407]]
        """
        lut = np.full([num_object_cls, num_action_cls], None)
        for hoi_id, object_id, action_id in class_corr:
            lut[object_id, action_id] = hoi_id
        return lut.tolist()

def swig_object_to_interaction(num_object_cls, _class_corr):
        """
        class_corr: List[(x["id"], x["object_id"], x["action_id"])]
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(num_object_cls)]
        for hoi_id, object_id, action_id in _class_corr:
            obj_to_int[object_id].append(hoi_id)
        return obj_to_int

def swig_object_to_verb(num_object_cls, _class_corr):
        """
        class_corr: List[(x["id"], x["object_id"], x["action_id"])]
        
        Returns:
            list[list]
        """
        obj_to_verb = [[] for _ in range(num_object_cls)]
        for hoi_id, object_id, action_id in _class_corr:
            obj_to_verb[object_id].append(action_id)
        return obj_to_verb

def swig_verb2interaction(num_action_cls, num_interaction_cls, class_corr):
    '''
    Returns: List[hoi_id] = action_id
    '''
    v2i = np.full([num_interaction_cls], None)
    for hoi_id, object_id, action_id in class_corr:
        v2i[hoi_id] = action_id
    return v2i.tolist()

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    # args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    # if args.clip_model_name == 'ViT-B-16':
    #     args.clip_model_name = 'ViT-B/16'
    # elif args.clip_model_name == 'ViT-L-14-336px':
    #     args.clip_model_name = 'ViT-L/14@336px'

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root, clip_model_name='ViT-B/16', zero_shot=args.zs, zs_type=args.zs_type, num_classes=args.num_classes)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root, clip_model_name='ViT-B/16')
    verb2interaction = None
    # trainset[0][1]: dict_keys(['boxes_h', 'boxes_o', 'hoi', 'object', 'verb', 'orig_size', 'labels', 'size', 'filename'])
    # trainset[0][0]: (torch.Size([3, 814, 640]), torch.Size([3, 224, 224]))
    if args.dataset == 'vcoco':
        class_corr = vcoco_class_corr()
        trainset.dataset.class_corr = class_corr
        testset.dataset.class_corr = class_corr
        object_n_verb_to_interaction = vcoco_object_n_verb_to_interaction(num_object_cls=len(trainset.dataset.objects), num_action_cls=len(trainset.dataset.actions), class_corr=class_corr)
        trainset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
        testset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction

    # args.hoi_cooccurence = tranverse_and_get_hoi_cooccurence(trainset.dataset)
    if args.training_set_ratio < 0.9:
        print(f'[INFO]: using {args.training_set_ratio} trainset to train!')
        sub_trainset, valset = trainset.dataset.split(args.training_set_ratio)
        trainset.dataset = sub_trainset
        trainset.keep = [i for i in range(len(sub_trainset))]
        
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )

    ## select pic for teaser
    # for imgs, tgts in test_loader:
    #     img, img_crop = imgs[0] ## batchsize==1
    #     tgt = tgts[0]
    #     length = len(tgt['hoi'])
    #     flag = True
    #     for i in range(length):
    #         for j in range(i+1, length):
    #             if box_iou(box_ops.box_cxcywh_to_xyxy(tgt['boxes_h'][i:i+1]), box_ops.box_cxcywh_to_xyxy(tgt['boxes_h'][j:j+1]))[0] < 0.8:
    #                 flag = False
    #     if flag:
    #         pdb.set_trace()
    # for imgs, tgts in test_loader:
    #     img, img_crop = imgs[0] ## batchsize==1
    #     tgt = tgts[0]
    #     if '51610' in tgt['filename']:
    #         pdb.set_trace() 

    # test_loader_of_trainingset = DataLoader(
    #     dataset=trainset,
    #     collate_fn=custom_collate, batch_size=1,
    #     num_workers=args.num_workers, pin_memory=False, drop_last=False,
    #     sampler=torch.utils.data.SequentialSampler(trainset)
    # )

    args.human_idx = 0
    if args.dataset == 'swig':
        object_n_verb_to_interaction = train_loader.dataset.object_n_verb_to_interaction
    else:
        object_n_verb_to_interaction = train_loader.dataset.dataset.object_n_verb_to_interaction

    if args.dataset == 'hicodet':
        if args.num_classes == 117:
            object_to_target = train_loader.dataset.dataset.object_to_verb
        elif args.num_classes == 600:
            object_to_target = train_loader.dataset.dataset.object_to_interaction
        
        if args.zs:
            object_to_target = train_loader.dataset.zs_object_to_target
    elif args.dataset == 'vcoco':
        if args.num_classes == 24:
            object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        elif args.num_classes == 236:
            raise NotImplementedError
    
    print('[INFO]: num_classes', args.num_classes)
    if args.dataset == 'vcoco' or args.dataset == 'swig':
        num_anno = None
    else:
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        if args.num_classes == 117:
            num_anno = torch.as_tensor(trainset.dataset.anno_action)
    upt = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction, clip_model_path=args.clip_dir_vit, num_anno=num_anno, verb2interaction=verb2interaction)
    if args.dataset == 'hicodet' and args.eval:  ## after building model, manually change obj_to_target
        if args.num_classes == 117:
            upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_verb
        else:
            upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_interaction
    if args.pseudo_label:  ## if we generate pseudo label for unseen verbs,
        pdb.set_trace()
        upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_verb

    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.ms:
            ## remove all params with 'clip_head' in the name
            for k in list(checkpoint['model_state_dict'].keys()):
                if 'clip_head' in k:
                    del checkpoint['model_state_dict'][k]
            ## load params for clip_head.image_encoder
            ms.load_checkpoint('./checkpoints/ms_image_encoder.ckpt', upt.clip_head.image_encoder)

        missing_keys, unexpected_keys = upt.load_state_dict(checkpoint['model_state_dict'], False)
        print(f"missing_keys: {missing_keys}", f"unexpected_keys: {unexpected_keys}")
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")
    
    if args.zs and args.fill_zs_verb_type == 1:
        upt.refresh_unseen_verb_cache_mem() ## whether refresh unseen weights after loading weights (during test)
    
    engine = CustomisedDLE(
        upt, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
    )
    if args.vis_tor != 1 and (args.eval or args.cache):
        upt.logit_scale_HO = torch.nn.Parameter(upt.logit_scale_HO * args.vis_tor)
        upt.logit_scale_U = torch.nn.Parameter(upt.logit_scale_U * args.vis_tor)

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            # print("[!NOTE!]: using test_loader_of_trainingset")
            engine.cache_vcoco(test_loader, args.output_dir)
        return
    
    if args.eval:
        device = torch.device(args.device)
        upt.eval()
        if args.dataset == 'vcoco':
            raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        ap = engine.test_hico(test_loader, args)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        print(
            f"The mAP is {ap.mean()*100:.2f},"
            f" rare: {ap[rare].mean()*100:.2f},"
            f" none-rare: {ap[non_rare].mean()*100:.2f},"
        )
        if args.zs:
            zs_hoi_idx = hico_unseen_index[args.zs_type]
            print(f'>>> zero-shot setting({args.zs_type}!!)')
            ap_unseen = []
            ap_seen = []
            for i, value in enumerate(ap):
                if i in zs_hoi_idx: 
                    ap_unseen.append(value)
                else: 
                    ap_seen.append(value)
            ap_unseen = torch.as_tensor(ap_unseen).mean()
            ap_seen = torch.as_tensor(ap_seen).mean()
            print(
                f"full mAP: {ap.mean()*100:.2f}",
                f"unseen: {ap_unseen*100:.2f}",
                f"seen: {ap_seen*100:.2f}",
            )
            
        return

    for p in upt.detector.parameters():
        p.requires_grad = False

    for n, p in upt.clip_head.named_parameters():
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj'): 
            p.requires_grad = True
            # print(n)
        elif 'adaptermlp' in n or "prompt_learner" in n:
            p.requires_grad = True
            # print(n) 
        else: p.requires_grad = False
    
    if args.frozen_classifier != None:
        frozen_name_lst = []
        if 'HO' in args.frozen_classifier:
            frozen_name_lst.append('adapter_HO')
        if 'U' in args.frozen_classifier:
            frozen_name_lst.append('adapter_U')
        if 'T' in args.frozen_classifier:
            frozen_name_lst.append('adapter_union')
        
        for n, p in upt.named_parameters():
            if 'clip_head' in n or 'detector' in n:
                continue
            if n.split('.')[0] in frozen_name_lst:
                p.requires_grad = False
          
    others = [n for n, p in upt.named_parameters()
                    if p.requires_grad and 'clip_head' not in n]

    param_dicts = [
        {
            "params": [p for n, p in upt.clip_head.named_parameters()
                    if p.requires_grad]
        },
        { ## others
            "params": [p for n, p in upt.named_parameters()
                    if p.requires_grad and 'clip_head' not in n],
            "lr": args.lr_head,
        },
    ]
    
    # print([n for n, p in upt.named_parameters()
    #     if p.requires_grad])
    if args.ms:
        import mindspore
        from mindspore.experimental import optim
        optimizer = optim.AdamW(
            param_dicts, lr=args.lr_vit,
            weight_decay=args.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        n_parameters = sum(p.numel() for p in upt.parameters() if p.requires_grad)
        print('number of leanable params:', n_parameters)
        
        n_parameters = sum(p.numel() for p in upt.parameters())
        print('number of all params:', n_parameters)

        optim = torch.optim.AdamW(
            param_dicts, lr=args.lr_vit,
            weight_decay=args.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    # Override optimiser and learning rate scheduler
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    # with torch.autograd.set_detect_anomaly(True):

    import json
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    # engine(args.epochs)
    ## Train for a specified number of epochs, do NOT use "engine"
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    not_use_scaler = True
    import pocket, time
    for epoch in range(args.epochs):
        print(f"===>>> Rank {rank}: start training epoch {epoch}")
        train_loader.sampler.set_epoch(epoch) # Update random seeds for sampler
        upt.train()
        if args.ms:
            def forward_fn(data, label):
                loss, logits = upt(data, label)
                return loss, logits
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            def train_step(data, label):
                (loss, _), grads = grad_fn(data, label)
                optimizer(grads)
                return loss
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = pocket.ops.relocate_to_cuda(imgs, non_blocking=True)
            targets = pocket.ops.relocate_to_cuda(targets, non_blocking=True)
            start_time = None
            if args.ms:
                losses = train_step(data=imgs, label=targets)
                # scheduler.step()
            else:
                loss_dict = upt(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                if not torch.isfinite(losses).item():
                    print(f"===>>> Rank {rank}: Loss is {losses}, stop training")
                    continue
                optim.zero_grad(set_to_none=True)
                losses.backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(upt.parameters(), args.clip_max_norm)
                optim.step()
            if i % args.print_interval == 0:
                print(f"===>>> Rank {rank}: epoch {epoch}, iteration {i}, loss: {losses}")
                if start_time is not None:
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    print(f"程序执行时间：{elapsed_time:.6f} 秒")
                start_time = time.perf_counter()
                break
        lr_scheduler.step() # same for ms and torch
        if rank == 0:
            pdb.set_trace()
            checkpoint = {
                'image_encoder': upt.clip_head.image_encoder,
            }
            if args.ms:
                ms.save_checkpoint(checkpoint, os.path.join(args.output_dir, f'ms_checkpoint_{epoch}.ckpt'))
            else:
                torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))


# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params

# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params

def param_convert(ms_params, pt_params, ckpt_path):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    for ms_param in ms_params.keys():
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数
        if "ln" in ms_param or "norm" in ms_param or "bn" in ms_param or "downsample.1" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)


@torch.no_grad()
def sanity_check(args):
    rank = 0
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )
    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(rank)

    # dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root, clip_model_name='ViT-B/16', zero_shot=args.zs, zs_type=args.zs_type, num_classes=args.num_classes)
    # testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root, clip_model_name='ViT-B/16')
    verb2interaction = None
    args.human_idx = 0
    object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction

    if args.dataset == 'hicodet':
        if args.num_classes == 117:
            object_to_target = trainset.dataset.object_to_verb
        elif args.num_classes == 600:
            object_to_target = trainset.dataset.object_to_interaction
        if args.zs:
            object_to_target = trainset.zs_object_to_target
    elif args.dataset == 'vcoco':
        object_to_target = list(trainset.dataset.object_to_action.values())
    
    print('[INFO]: num_classes', args.num_classes)
    num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
    if args.num_classes == 117:
        num_anno = torch.as_tensor(trainset.dataset.anno_action)
    upt = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction, clip_model_path=args.clip_dir_vit, num_anno=num_anno, verb2interaction=verb2interaction)
    if os.path.exists(args.resume):
        print(f"===>>> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.ms:
            ## remove all params with 'clip_head' in the name
            for k in list(checkpoint['model_state_dict'].keys()):
                if 'clip_head' in k:
                    del checkpoint['model_state_dict'][k]
            ## load params for clip_head.image_encoder
            param_dict = ms.load_checkpoint('./checkpoints/ms_image_encoder.ckpt', net=upt.clip_head.image_encoder)
        
        missing_keys, unexpected_keys = upt.load_state_dict(checkpoint['model_state_dict'], False)
        print("param_not_load for upt",f"missing_keys: {missing_keys}", f"unexpected_keys: {unexpected_keys}")
    if args.eval:
        upt.eval()

    ### convert upt.clip_head.image_encoder to onnx
    # images_clip = torch.randn(1, 3, 224, 224) # tensor([[[[-0.8696,  0.0776, -1.4397,  ...,  0.3938,  0.5942,  0.2643],
    # prior = torch.randn(1, 7, 64) # tensor([[[-1.6804e+00,  1.0004e+00, -5.1594e-01,  7.8591e-01,  2.1071e-01,
    # prior_mask = torch.zeros(1, 7) # tensor([[0., 0., 0., 0., 0., 0., 0.]])
    import pickle
    # with open('input.pkl', 'wb') as f:
    #     pickle.dump((images_clip, prior, prior_mask), f)
    with open('input.pkl', 'rb') as f:
        images_clip, prior, prior_mask = pickle.load(f)
    if args.ms:
        ms.set_context(device_target='GPU')
        images_clip = ms.Tensor(images_clip.numpy()) # tensor([[[[-0.8696,  0.0776, -1.4397,  ...,  0.3938,  0.5942,  0.2643],
        prior = ms.Tensor(prior.numpy())
        prior_mask = ms.Tensor(prior_mask.numpy())
    dummy_input = (images_clip, prior, prior_mask)
    feat_global, feat_local = upt.clip_head.image_encoder(*dummy_input)
    ## w/o model weights
    # pt_model: tensor([[ 5.6911e-01, -6.6572e-01, -2.1629e-01,  8.3021e-02, -3.0263e-01, ...
    # ms_model: [[ 5.69105685e-01, -6.65721536e-01, -2.16294542e-01 ...  2.98165381e-01, -1.70433551e-01,  5.33226967e-01]])
    # w/ model weights
    # pt_model: tensor([[ 3.3815e-01,  2.6248e-01, -3.7052e-01,  3.1337e-01, -1.7075e-01, ...
    # ms_model: [[ 5.69106102e-01, -6.65721357e-01, -2.16294408e-01 ...  2.98165202e-01, -1.70433730e-01,  5.33227146e-01]])
    
    import pdb; pdb.set_trace()
    # torch.save(upt.clip_head.image_encoder.state_dict(), './checkpoints/image_encoder.pth')
    pt_params = pytorch_params('./checkpoints/image_encoder.pth')
    ms_params = mindspore_params(upt.clip_head.image_encoder)

    new_pt_params = {}
    ## add prefix 'visual' to the key of pt_params
    for k, v in pt_params.items():
        new_pt_params['visual.'+k] = v
    param_convert(ms_params, new_pt_params, './checkpoints/ms_image_encoder.ckpt')
    exit()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-3, type=float)
    parser.add_argument('--lr-vit', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--ms', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--visual_mode', default='vit', type=str)
    # add CLIP model resenet 
    # parser.add_argument('--clip_dir', default='./checkpoints/pretrained_clip/RN50.pt', type=str)
    # parser.add_argument('--clip_visual_layers', default=[3, 4, 6, 3], type=list)
    # parser.add_argument('--clip_visual_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_visual_input_resolution', default=1344, type=int)
    # parser.add_argument('--clip_visual_width', default=64, type=int)
    # parser.add_argument('--clip_visual_patch_size', default=64, type=int)
    # parser.add_argument('--clip_text_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_text_transformer_width', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers', default=12, type=int)
    # parser.add_argument('--clip_text_context_length', default=13, type=int)

    #### add CLIP vision transformer
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)

    ### ViT-L/14@336px START: emb_dim: 768 
    # >>> vision_width: 1024,  vision_patch_size(conv's kernel-size&&stride-size): 14,
    # >>> vision_layers(#layers in vision-transformer): 24 ,  image_resolution:336;
    # >>> transformer_width:768, transformer_layers: 12, transformer_heads:12
    parser.add_argument('--clip_visual_layers_vit', default=24, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)

    # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-L/14@336px----END----
    
    ### ViT-B-16 START
    # parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    # parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    # parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    # parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int)

    # # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-B-16-----END-----
    parser.add_argument('--clip_text_context_length_vit', default=77, type=int) # 13 -77
    parser.add_argument('--use_insadapter', action='store_true')
    parser.add_argument('--use_distill', action='store_true')
    parser.add_argument('--use_consistloss', action='store_true')
    
    parser.add_argument('--use_mean', action='store_true') # 13 -77
    parser.add_argument('--logits_type', default='HO+U+T', type=str) # 13 -77 # text_add_visual, visual
    parser.add_argument('--num_shot', default='2', type=int) # 13 -77 # text_add_visual, visual
    parser.add_argument('--file1', default='./hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p',type=str)
    parser.add_argument('--prior_type', type=str, default='cbe', choices=['cbe', 'cb', 'ce', 'be', 'c', 'b', 'e'])
    parser.add_argument('--obj_affordance', action='store_true') ## use affordance embedding of objects
    parser.add_argument('--training_set_ratio', type=float, default=1.0)
    parser.add_argument('--frozen_classifier', type=str, default=None)
    parser.add_argument('--zs', action='store_true') ## zero-shot
    parser.add_argument('--hyper_lambda', type=float, default=2.8)
    parser.add_argument('--use_weight_pred', action='store_true')
    parser.add_argument('--zs_type', type=str, default='rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4'])
    parser.add_argument('--fill_zs_verb_type', type=int, default=0,) # (for init) 0: random; 1: weighted_sum, 
    parser.add_argument('--pseudo_label', action='store_true') 
    parser.add_argument('--tpt', action='store_true') 
    parser.add_argument('--vis_tor', type=float, default=1.0)
    parser.add_argument('--adapter_num_layers', type=int, default=1)

    ## prompt learning
    parser.add_argument('--N_CTX', type=int, default=24)  # number of context vectors
    parser.add_argument('--CSC', type=bool, default=False)  # class-specific context
    parser.add_argument('--CTX_INIT', type=str, default='')  # initialization words
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end')  # # 'middle' or 'end' or 'front'

    parser.add_argument('--use_templates', action='store_true') 
    parser.add_argument('--LA', action='store_true')  ## Language Aware
    parser.add_argument('--LA_weight', default=0.6, type=float)  ## Language Aware(loss weight)

    parser.add_argument('--feat_mask_type', type=int, default=0,) # 0: dropout(random mask); 1: None
    parser.add_argument('--num_classes', type=int, default=117,) 
    parser.add_argument('--prior_method', type=int, default=0) ## 0: instance-wise, 1: pair-wise, 2: learnable
    parser.add_argument('--vis_prompt_num', type=int, default=50) ##  (prior_method == learnable)
    parser.add_argument('--box_proj', type=int, default=0,) ## 0: None; 1: f_u = ROI-feat + MLP(uni-box)
    parser.add_argument('--adapter_pos', type=str, default='all', choices=['all', 'front', 'end', 'random', 'last'])
    parser.add_argument('--use_multi_hot', action='store_true')
    parser.add_argument('--label_learning', action='store_true')
    parser.add_argument('--label_choice', default='random', choices=['random', 'single_first', 'multi_first', 'single+multi', 'rare_first', 'non_rare_first', 'rare+non_rare'])
    parser.add_argument('--use_mlp_proj', action='store_true')
    
    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')
    
    ## **************** arguments for deformable detr **************** ##
    parser.add_argument('--d_detr', default=False, type=lambda x: (str(x).lower() == 'true'),)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    ## **************** arguments for deformable detr **************** ##
    args = parser.parse_args()
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    if args.sanity:
        sanity_check(args)
        sys.exit()
    # mp.spawn(main, nprocs=args.world_size, args=(args,))
    if args.world_size==1:
        main(0,args)
    else:
        mp.spawn(main, nprocs=args.world_size, args=(args,))
