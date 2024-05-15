"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


from builtins import Exception
import os
import torch
import torch.distributed as dist
import mindspore as ms
import mindspore.ops.operations as P
from torch import nn
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import binary_focal_loss_with_logits, binary_focal_loss_with_logits_ms

import sys
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,hico_verbs_sentence_2
from vcoco_list import vcoco_verbs_sentence
sys.path.append('detr')
# print(sys.path)
from detr.models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
import pdb
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from transformer_module import TransformerDecoderLayer
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip 
from ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import pickle, random
from tqdm import tqdm
from hico_text_label import hico_unseen_index
import hico_text_label

_tokenizer = _Tokenizer()
class MLP_ms(ms.nn.Cell):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ms.nn.CellList(list(ms.nn.Dense(in_channels=n, out_channels=k) for n, k in zip([input_dim] + h, h + [output_dim])))

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = P.ReLU()(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Weight_Pred(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear1 = MLP(input_dim=input_dim, hidden_dim=512, output_dim=128, num_layers=2)
        self.drop1 = nn.Dropout()
        self.linear2 = MLP(input_dim=128, hidden_dim=32, output_dim=3, num_layers=2)
    
    def forward(self, x):
        x = self.drop1(self.linear1(x))
        x = self.linear2(x)
        return F.sigmoid(x)

class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        # self.prompt_learner = PromptLearner(args, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        # self.text_encoder = TextEncoder(clip_model)
        # self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        return 0


class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module 
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action/interaction classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        args,
        detector: nn.Module,
        postprocessor: nn.Module,
        model: nn.Module,
        origin_text_embeddings: torch.tensor,
        object_embedding: torch.tensor,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model
        self.origin_text_embeddings = origin_text_embeddings
        self.object_embedding = object_embedding
        self.visual_output_dim = model.image_encoder.output_dim
        self.object_n_verb_to_interaction = np.asarray(
                                object_n_verb_to_interaction, dtype=float
                            )

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self.num_anno = kwargs["num_anno"]

        self.use_distill = args.use_distill
        self.use_consistloss = args.use_consistloss

        self.num_classes = num_classes
        self.use_multi_hot = args.use_multi_hot

        self.feature = []
        if 'HO' in args.logits_type:
            self.feature.append('hum_obj')
        if 'U' in args.logits_type or 'T' in args.logits_type:
            self.feature.append('uni')
        self.feature = '_'.join(self.feature)
        self.logits_type = args.logits_type

        num_shot = args.num_shot
        file1 = args.file1

        if args.zs:
            self.zs_type = args.zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx = []
            self.zs_type = None

        self.unseen_verb_idxs = []
        self.label_choice = args.label_choice
        if 'HO' in self.logits_type:
            self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO = self.load_cache_model(file1=file1, feature='hum_obj',num_classes=self.num_classes, num_shot=num_shot, filtered_hoi_idx = self.filtered_hoi_idx, use_multi_hot=self.use_multi_hot, label_choice=self.label_choice, num_anno=self.num_anno)
            if not args.sanity:
                self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO  = self.cache_model_HO.cuda().float(), self.one_hots_HO.cuda().float(), self.sample_lens_HO.cuda().float()
        if 'U' in self.logits_type:
            self.cache_model_U, self.one_hots_U, self.sample_lens_U = self.load_cache_model(file1=file1, feature='uni',num_classes=self.num_classes, num_shot=num_shot, filtered_hoi_idx = self.filtered_hoi_idx, use_multi_hot=self.use_multi_hot, label_choice=self.label_choice, num_anno=self.num_anno)
            if not args.sanity:
                self.cache_model_U, self.one_hots_U, self.sample_lens_U = self.cache_model_U.cuda().float(), self.one_hots_U.cuda().float(), self.sample_lens_U.cuda().float()

        if self.num_classes == 117:
            self.seen_verb_idxs = [i for i in range(self.num_classes) if i not in self.unseen_verb_idxs]
        elif self.num_classes == 600:
            self.seen_hoi_idxs = [i for i in range(self.num_classes) if i not in self.filtered_hoi_idx]
        
        self.individual_norm = True
        self.logits_type = args.logits_type #
        self.consist = True
        self.evaluate_type = 'detr' # gt, detr
        
        self.use_type = 'crop'
        self.beta_cache = torch.tensor(10)
        self.alpha_cache = torch.tensor(1.0)

        self.prior_type = args.prior_type
        self.finetune_adapter = True
        if self.prior_type == 'cbe':
            self.priors_initial_dim = self.visual_output_dim+5
        elif self.prior_type == 'cb':
            self.priors_initial_dim = 5
        elif self.prior_type == 'ce':
            self.priors_initial_dim = self.visual_output_dim+1
        elif self.prior_type == 'be':
            self.priors_initial_dim = self.visual_output_dim+4
        elif self.prior_type == 'c':
            self.priors_initial_dim = 1
        elif self.prior_type == 'b':
            self.priors_initial_dim = 4
        elif self.prior_type == 'e':
            self.priors_initial_dim = self.visual_output_dim

        self.use_weight_pred = args.use_weight_pred
        # if args.ms:
        #     ms.set_context(device_target='GPU')
        #     self.sample_lens_HO = ms.Tensor(self.sample_lens_HO.cpu().numpy(),)
        #     self.sample_lens_U = ms.Tensor(self.sample_lens_U.cpu().numpy(),)
        if self.finetune_adapter:
            # if args.ms:
            #     self.matmul = ms.ops.MatMul()
            #     if 'HO' in self.logits_type:
            #         self.adapter_HO_weight = ms.Parameter(ms.Tensor(self.cache_model_HO.clone().detach().cpu().numpy()))
            #         self.adapter_HO_bias = ms.Parameter(ms.Tensor(-np.ones(self.cache_model_HO.shape[0]))).float()
            #         self.label_HO = ms.Parameter(ms.Tensor(self.one_hots_HO.cpu().numpy()), requires_grad=False).float()
            #         if not self.use_weight_pred:
            #             self.logit_scale_HO = ms.Parameter(ms.Tensor(np.log(1 / 0.07)).float())
            #     if 'U' in self.logits_type:
            #         self.adapter_U_weight = ms.Parameter(ms.Tensor(self.cache_model_U.clone().detach().cpu().numpy()))
            #         self.adapter_U_bias = ms.Parameter(ms.Tensor(-np.ones(self.cache_model_U.shape[0]))).float()
            #         self.label_U = ms.Parameter(ms.Tensor(self.one_hots_U.cpu().numpy()), requires_grad=False).float()
            #         if not self.use_weight_pred:
            #             self.logit_scale_U = ms.Parameter(ms.Tensor(np.log(1 / 0.07)).float())
            #     if 'T' in self.logits_type:
            #         self.adapter_union_weight = ms.Parameter(ms.Tensor(self.origin_text_embeddings.clone().detach().cpu().numpy()))
            #         if not self.use_weight_pred:
            #             self.logit_scale_text = ms.Parameter(ms.Tensor(np.log(1 / 0.07)).float())
            # else:
            if 'HO' in self.logits_type:
                self.adapter_HO_weight = nn.Parameter(self.cache_model_HO.clone().detach())
                self.adapter_HO_bias = nn.Parameter(-torch.ones(self.cache_model_HO.shape[0]))
                self.label_HO = nn.Parameter(self.one_hots_HO, requires_grad=False).float()
                if not self.use_weight_pred:
                    self.logit_scale_HO = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).float()
            if 'U' in self.logits_type:
                self.adapter_U_weight = nn.Parameter(self.cache_model_U.clone().detach())
                self.adapter_U_bias = nn.Parameter(-torch.ones(self.cache_model_U.shape[0]))
                self.label_U = nn.Parameter(self.one_hots_U, requires_grad=False).float()
                if not self.use_weight_pred:
                    self.logit_scale_U = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).float()
            if 'T' in self.logits_type:
                self.adapter_union_weight = nn.Parameter(self.origin_text_embeddings.clone().detach())
                if not self.use_weight_pred:
                    self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).float()
        
        if args.use_insadapter:
            assert args.prior_method == 0, "args.prior_method != 0 !!"
            # if args.ms:
            #     self.priors_downproj = MLP_ms(self.priors_initial_dim, 128, 64, 3) # old 512+5
            # else:
            self.priors_downproj = MLP(self.priors_initial_dim, 128, 64, 3) # old 512+5   

        self.no_interaction_indexes = [9, 23, 30, 45, 53, 64, 75, 85, 91, 95, 106, 110, 128, 145, 159, 169, 173, 185, 193, 197, 207, 213, 223, 231, 234, 238, 242, 246, 251, 256, 263, 272, 282, 289, 294, 304, 312, 324, 329, 335, 341, 347, 351, 355, 362, 367, 375, 382, 388, 392, 396, 406, 413, 417, 428, 433, 437, 444, 448, 452, 462, 473, 482, 487, 501, 505, 515, 527, 532, 537, 545, 549, 557, 561, 566, 575, 583, 587, 594, 599]
        self.HOI_IDX_TO_OBJ_IDX = [
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14,
                14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 39,
                39, 39, 39, 39, 39, 39, 39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 56, 56, 56, 56,
                56, 56, 57, 57, 57, 57, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 60, 60,
                60, 60, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58,
                58, 58, 58, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 6, 6, 6, 6, 6,
                6, 6, 6, 62, 62, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 24, 24,
                24, 24, 24, 24, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 34, 34, 34, 34,
                34, 34, 34, 35, 35, 35, 21, 21, 21, 21, 59, 59, 59, 59, 13, 13, 13, 13, 73,
                73, 73, 73, 73, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 67, 67, 67,
                67, 67, 67, 67, 74, 74, 74, 74, 74, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                54, 54, 54, 54, 54, 54, 54, 54, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                20, 10, 10, 10, 10, 10, 42, 42, 42, 42, 42, 42, 29, 29, 29, 29, 29, 29, 23,
                23, 23, 23, 23, 23, 78, 78, 78, 78, 26, 26, 26, 26, 52, 52, 52, 52, 52, 52,
                52, 66, 66, 66, 66, 66, 33, 33, 33, 33, 33, 33, 33, 33, 43, 43, 43, 43, 43,
                43, 43, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 64, 64, 64, 64, 49, 49, 49,
                49, 49, 49, 49, 49, 49, 49, 69, 69, 69, 69, 69, 69, 69, 12, 12, 12, 12, 53,
                53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 72, 72, 72, 72, 72, 65, 65, 65, 65,
                48, 48, 48, 48, 48, 48, 48, 76, 76, 76, 76, 71, 71, 71, 71, 36, 36, 36, 36,
                36, 36, 36, 36, 36, 36, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 44, 44, 44, 44, 44, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 11, 11, 11, 11, 28, 28, 28, 28, 28, 28, 28, 28,
                28, 28, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 77, 77, 77, 77, 77,
                38, 38, 38, 38, 38, 27, 27, 27, 27, 27, 27, 27, 27, 70, 70, 70, 70, 61, 61,
                61, 61, 61, 61, 61, 61, 79, 79, 79, 79, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 25, 25, 25, 25, 25, 25, 25, 25, 75, 75, 75, 75, 40, 40, 40, 40, 40,
                40, 40, 22, 22, 22, 22, 22
            ]
        self.obj_to_no_interaction = torch.as_tensor([169, 23, 75, 159, 9, 64, 193, 575, 45, 566, 329, 505, 417, 246,
                                                        30,  85, 128, 145, 185, 106, 324, 238, 599, 347, 213, 583, 355, 545,
                                                        515, 341, 473, 482, 501, 375, 231, 234, 462, 527, 537,  53, 594, 304,
                                                        335, 382, 487, 256, 223, 207, 444, 406, 263, 282, 362, 428, 312, 272,
                                                        91,  95, 173, 242, 110, 557, 197, 388, 396, 437, 367, 289, 392, 413,
                                                        549, 452, 433, 251, 294, 587, 448, 532, 351, 561])

        self.epoch = 0
        # self.use_deformable_attn = args.use_deformable_attn
        self.COCO_CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
                    'fire hydrant','N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
                    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', \
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', \
                    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', \
                    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
                    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.reserve_indices = [idx for (idx, name) in enumerate(self.COCO_CLASSES) if name != 'N/A']
        self.reserve_indices = self.reserve_indices + [91]
        self.reserve_indices = torch.as_tensor(self.reserve_indices)
        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda
        self.pseudo_label = args.pseudo_label
        self.tpt = args.tpt
        self.featmap_dropout = nn.Dropout(0.2)
        self.feat_mask_type = args.feat_mask_type
        self.language_aware = args.LA 
        self.use_insadapter = args.use_insadapter
        self.prior_method = args.prior_method
        self.LA_weight = args.LA_weight
        self.box_proj = args.box_proj
        if self.box_proj:
            self.box_proj_mlp = MLP(8, 128, self.visual_output_dim, num_layers=3)
        if self.use_weight_pred:
            num_branch = len(self.logits_type.split('+'))
            self.weight_pred = Weight_Pred(input_dim=self.visual_output_dim*3, output_dim=num_branch)
        
        self.use_mlp_proj = kwargs["use_mlp_proj"]
        if self.use_mlp_proj:
            self.mlp_proj = MLP(512, 512, 512, 3)
        self.ms = args.ms
        # if self.ms:
        #     self.object_embedding = ms.Tensor(self.object_embedding.numpy())

    def load_cache_model(self,file1, feature='uni',num_classes=117, num_shot=10, filtered_hoi_idx=[], use_multi_hot=False, label_choice='random', num_anno=None):  ## √
        annotation = pickle.load(open(file1,'rb'))
        categories = num_classes
        union_embeddings = [[] for i in range(categories)]
        obj_embeddings = [[] for i in range(categories)]
        hum_embeddings = [[] for i in range(categories)]
        real_verbs = [[] for i in range(categories)]
        filenames = list(annotation.keys())
        verbs_iou = [[] for i in range(categories)] # contain 600hois or 117 verbs
        # hois_iou = [[] for i in range(len(hois))]
        each_filenames = [[] for i in range(categories)]
        for file_n in filenames:
            anno = annotation[file_n]
            # dict_keys (['boxes_h', 'boxes_o', 'verbs', 'union_boxes', 'union_features', 'huamn_features''object_features''objects', 'global_feature'])
            if categories == 117 or categories == 24: verbs = anno['verbs']
            else: verbs = (self.object_n_verb_to_interaction[anno['objects'], anno['verbs']]).astype(int)
            
            num_ho_pair = len(anno['boxes_h'])
            anno['real_verbs'] = np.zeros(shape=(num_ho_pair, categories))
            for i in range(num_ho_pair):
                anno['real_verbs'][i][verbs[i]] = 1

            if use_multi_hot:
                boxes_h_iou = torchvision.ops.box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_h']))
                boxes_o_iou = torchvision.ops.box_iou(torch.as_tensor(anno['boxes_o']), torch.as_tensor(anno['boxes_o']))
                for i in range(num_ho_pair):
                    idx_h = boxes_h_iou[i] > 0.6
                    idx_o = torch.logical_and(boxes_o_iou[i] > 0.6, torch.as_tensor(anno['objects']) == anno['objects'][i])
                    idx_ho = torch.logical_and(idx_h, idx_o)
                    anno['real_verbs'][i] = torch.sum(torch.as_tensor(anno['real_verbs'])[idx_ho], dim=0)
                
                anno['real_verbs'][anno['real_verbs']>1] = 1

            ious = torch.diag(box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_o'])))
            if len(verbs) == 0:
                print(file_n)

            for i, v in enumerate(verbs):
                if 'hico' in file1: ## TODO ??? why vcoco list idx out of range
                    if num_classes == 117:
                        if anno['verbs'][i] not in self.object_class_to_target_class[anno['objects'][i]]:
                            continue
                    elif num_classes == 600:
                        if v in filtered_hoi_idx:
                            continue
                union_embeddings[v].append(anno['union_features'][i] / np.linalg.norm(anno['union_features'][i]))
                obj_embeddings[v].append(anno['object_features'][i] / np.linalg.norm(anno['object_features'][i]))
                hum_embeddings[v].append(anno['huamn_features'][i] / np.linalg.norm(anno['huamn_features'][i]))
                each_filenames[v].append(file_n)
                real_verbs[v].append(anno['real_verbs'][i])
                # add iou
                verbs_iou[v].append(ious[i])

        if num_classes == 117:
            for i in range(categories):
                if len(union_embeddings[i]) == 0:
                    self.unseen_verb_idxs.append(i)
            print('[INFO]: missing idxs of verbs:', self.unseen_verb_idxs)
            for i in self.unseen_verb_idxs:
                for z in range(num_shot):
                    union_embeddings[i].append(np.random.randn(self.visual_output_dim))
                    obj_embeddings[i].append(np.random.randn(self.visual_output_dim))
                    hum_embeddings[i].append(np.random.randn(self.visual_output_dim))
        elif num_classes == 600:
            for i in filtered_hoi_idx:
                for z in range(num_shot):
                    union_embeddings[i].append(np.random.randn(self.visual_output_dim))
                    obj_embeddings[i].append(np.random.randn(self.visual_output_dim))
                    hum_embeddings[i].append(np.random.randn(self.visual_output_dim))
        ## re-implement cachemodel construction
        cache_models_lst, each_lens_lst = [], []
        real_verbs_lst = []
        if feature == 'hum_obj':
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings, real_v in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings, real_verbs)):
                hum_emb =  torch.as_tensor(np.array(hum_emb)).float()   
                obj_emb = torch.as_tensor(np.array(obj_emb)).float()
                real_v = torch.as_tensor(np.array(real_v))
                new_embeddings = torch.cat([hum_emb, obj_emb], dim=-1)
                new_embeddings = new_embeddings.cuda().float()

                num_to_select = min(hum_emb.shape[0], num_shot)
                
                if num_to_select < hum_emb.shape[0]:
                    if label_choice == 'random':
                        topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                    elif label_choice == 'multi_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select)
                    elif label_choice == 'single_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select, largest=False)
                    elif label_choice == 'single+multi':
                        v_, topk_idx1 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    elif label_choice == 'rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=False)
                    elif label_choice == 'non_rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=True)
                    elif label_choice == 'rare+non_rare':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx1 = torch.topk(real_freq, k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(real_freq, k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    new_embeddings = new_embeddings[topk_idx]
                    real_v = real_v[topk_idx]
                
                cache_models_lst.append(new_embeddings)
                each_lens_lst.append(num_to_select)
                real_verbs_lst.append(real_v)
        elif feature == 'uni':
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings, real_v in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings, real_verbs)):
                uni_emb = torch.as_tensor(np.array(embeddings)).float()
                real_v = torch.as_tensor(np.array(real_v))
                new_embeddings = uni_emb
                new_embeddings = new_embeddings.cuda().float()

                num_to_select = min(uni_emb.shape[0], num_shot)

                if num_to_select < uni_emb.shape[0]:
                    if label_choice == 'random':
                        topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                    elif label_choice == 'multi_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select)
                    elif label_choice == 'single_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select, largest=False)
                    elif label_choice == 'single+multi':
                        v_, topk_idx1 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    elif label_choice == 'rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=False)
                    elif label_choice == 'non_rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=True)
                    elif label_choice == 'rare+non_rare':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx1 = torch.topk(real_freq, k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(real_freq, k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    new_embeddings = new_embeddings[topk_idx]
                    real_v = real_v[topk_idx]
                cache_models_lst.append(new_embeddings)
                each_lens_lst.append(num_to_select)
                real_verbs_lst.append(real_v)
        
        cache_models = torch.cat(cache_models_lst, dim=0)
        labels = torch.cat(real_verbs_lst, dim=0)
        return cache_models, labels, torch.sum(labels, dim=0)

    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_prior_scores(self,
        x: torch.Tensor, y: torch.Tensor, scores: torch.Tensor, object_class: torch.Tensor
    ) -> torch.Tensor:  ### √
        
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
        
        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)
        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping 
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def compute_roi_embeddings(self, features: OrderedDict, image_size: torch.Tensor, region_props: List[dict]):
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []

        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        gt_feats_collated = []
        pair_feats_collated = []
        gt_all_logits = []
        pair_logits = []
        pair_prior = []
        gt_labels = []
        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                # raise ValueError("There are no valid human-object pairs")
                return 0

            x = x.flatten(); y = y.flatten()
            
            # extract single roi features
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)
            
            spatial_scale = 1 / (image_size[0,0]/local_features.shape[1])
            # union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(1, 1),spatial_scale=spatial_scale,aligned=True).flatten(2).mean(-1)
            # single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes],output_size=(1, 1),spatial_scale=spatial_scale,aligned=True).flatten(2).mean(-1)
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)
            single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)

            if self.feat_mask_type == 0:
                union_features = self.featmap_dropout(union_features).flatten(2).mean(-1)
                single_features = self.featmap_dropout(single_features).flatten(2).mean(-1)
            elif self.feat_mask_type == 1:
                union_features = union_features.flatten(2).mean(-1)
                single_features = single_features.flatten(2).mean(-1)

            human_features = single_features[x_keep]
            object_features = single_features[y_keep]

            concat_feat_original = torch.cat([human_features,object_features, union_features],dim=-1)
            human_features = human_features / human_features.norm(dim=-1, keepdim=True)
            object_features = object_features / object_features.norm(dim=-1, keepdim=True)
            union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            if self.feature == 'hum_obj_uni':
                concat_feat = torch.cat([human_features, object_features, union_features],dim=-1) 
            elif self.feature == 'hum_obj':
                concat_feat = torch.cat([human_features, object_features], dim=-1)
            elif self.feature == 'uni':
                concat_feat = union_features

            if self.logits_type == 'HO+U+T':
                phi_union_HO = torch.cat([human_features, object_features], dim=-1) @ self.adapter_HO_weight.transpose(0, 1) + self.adapter_HO_bias
                phi_union_U = union_features @ self.adapter_U_weight.transpose(0, 1) + self.adapter_U_bias
                logits_cache_HO = ((phi_union_HO @ self.label_HO) / self.sample_lens_HO) / 2
                logits_cache_U = (phi_union_U @ self.label_U) / self.sample_lens_U
                logits_text = union_features @ self.adapter_union_weight.transpose(0, 1)
                
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_cache_HO * logits_weights[:, 0:1] + logits_cache_U * logits_weights[:, 1:2] + logits_text * logits_weights[:, 2:3]
                else:
                    logits = logits_cache_HO * self.logit_scale_HO + logits_cache_U * self.logit_scale_U + logits_text * self.logit_scale_text

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            all_logits.append(logits)

        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, gt_feats_collated, pair_feats_collated
    
    def compute_prior_scores_ms(self, x, y, scores, object_class):

        prior_h = ms.ops.zeros((len(x), self.num_classes), dtype=ms.float32)
        prior_o = ms.ops.zeros_like(prior_h)
        
        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)
        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping 
        target_cls_idx = []
        for obj in object_class[y]:
            object_class_to_target_class = self.object_class_to_target_class
            cur_lst = object_class_to_target_class[obj.item()]
            target_cls_idx.append(cur_lst)
        # target_cls_idx = [self.object_class_to_target_class[obj.item()] for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return ms.ops.stack([prior_h, prior_o])

    def compute_roi_embeddings_ms(self, features: OrderedDict, image_size: torch.Tensor, region_props: List[dict]):
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []

        img_h, img_w = image_size.unbind(-1)
        scale_fct = ms.ops.stack([img_w, img_h, img_w, img_h], axis=1)

        gt_feats_collated = []
        pair_feats_collated = []
        gt_all_logits = []
        pair_logits = []
        pair_prior = []
        gt_labels = []
        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']

            is_human = labels == self.human_idx
            n_h = ms.ops.sum(is_human); n = len(boxes)
            
            # Permute human instances to the top
            if not ms.ops.all(labels[:n_h]==self.human_idx):
                h_idx = ms.ops.nonzero(is_human).squeeze(1)
                o_idx = ms.ops.nonzero(is_human == 0).squeeze(1)
                perm = P.Concat()([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(ms.ops.zeros((0), dtype=ms.int64))
                boxes_o_collated.append(ms.ops.zeros((0), dtype=ms.int64))
                object_class_collated.append(ms.ops.zeros((0), dtype=ms.int64))
                prior_collated.append(ms.ops.zeros((2, 0, self.num_classes)))
                continue

            # Get the pairwise indices
            y, x = ms.ops.meshgrid(
                ms.ops.arange(n),
                ms.ops.arange(n)
            )
            # Valid human-object pairs
            x_keep, y_keep = ms.ops.nonzero(ms.ops.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                # raise ValueError("There are no valid human-object pairs")
                return 0
            
            x = P.Flatten()(x); y = P.Flatten()(y)
            
            # extract single roi features
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = ms.ops.minimum(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = ms.ops.maximum(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = P.Concat(-1)([lt, rb])
            
            spatial_scale = float(1 / (1.0*image_size[0,0]/local_features.shape[1]))
            # union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(1, 1),spatial_scale=spatial_scale,aligned=True).flatten(2).mean(-1)
            # single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes],output_size=(1, 1),spatial_scale=spatial_scale,aligned=True).flatten(2).mean(-1)
            # img_idx: (0,0, ...,0) shape:(union_boxes.shape[0], 1)
            img_idx = ms.ops.zeros((union_boxes.shape[0], 1))
            union_features = ms.ops.ROIAlign(7, 7, spatial_scale)(local_features.unsqueeze(0), ms.ops.cat((img_idx, union_boxes), axis=1))
            img_idx = ms.ops.zeros((boxes.shape[0], 1))
            single_features = ms.ops.ROIAlign(7, 7, spatial_scale)(local_features.unsqueeze(0), ms.ops.cat((img_idx, boxes), axis=1))
            
            if self.feat_mask_type == 0:
                union_features = ms.ops.flatten(ms.nn.Dropout(p=0.2)(union_features), start_dim=2).mean(-1)
                single_features = ms.ops.flatten(ms.nn.Dropout(p=0.2)(single_features), start_dim=2).mean(-1)
            elif self.feat_mask_type == 1:
                union_features = P.ReduceMean()(P.Flatten()(union_features), -1)
                single_features = P.ReduceMean()(P.Flatten()(single_features), -1)

            human_features = single_features[x_keep]
            object_features = single_features[y_keep]

            concat_feat_original = P.Concat(-1)([human_features, object_features, union_features])
            human_features = human_features / human_features.norm(dim=-1, keepdim=True)
            object_features = object_features / object_features.norm(dim=-1, keepdim=True)
            union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            if self.feature == 'hum_obj_uni':
                concat_feat = P.Concat(-1)([human_features, object_features, union_features]) 
            elif self.feature == 'hum_obj':
                concat_feat = P.Concat(-1)([human_features, object_features])
            elif self.feature == 'uni':
                concat_feat = union_features

            if self.logits_type == 'HO+U+T':
                phi_union_HO = self.matmul(ms.ops.cat([human_features, object_features], axis=-1), self.adapter_HO_weight.T) + self.adapter_HO_bias
                phi_union_U = self.matmul(union_features, self.adapter_U_weight.T) + self.adapter_U_bias
                logits_cache_HO = (self.matmul(phi_union_HO, self.label_HO) / self.sample_lens_HO) / 2
                logits_cache_U = self.matmul(phi_union_U, self.label_U) / self.sample_lens_U
                logits_text = self.matmul(union_features, self.adapter_union_weight.T)
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(ms.ops.cat([human_features, object_features, union_features], axis=-1))
                    logits = logits_cache_HO * logits_weights[:, 0:1] + logits_cache_U * logits_weights[:, 1:2] + logits_text * logits_weights[:, 2:3]
                else:
                    logits = logits_cache_HO * self.logit_scale_HO + logits_cache_U * self.logit_scale_U + logits_text * self.logit_scale_text
            
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores_ms(
                x_keep, y_keep, scores, labels)
            )
            all_logits.append(logits)

        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, gt_feats_collated, pair_feats_collated
        
    def recover_boxes(self, boxes, size):  
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets): ## for training
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        
        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
        # print("pair gt,",len(x),len(y))
        # IndexError: tensors used as indices must be long, byte or bool tensors
        if self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1  ## target['labels']: verb/action
        else:
            labels[x, targets['hoi'][y]] = 1
        # print("#(labels==1) = ", torch.sum(labels))
        return labels

    def associate_with_ground_truth_ms(self, boxes_h, boxes_o, targets):
        device = targets['boxes_h'].device
        n = boxes_h.shape[0]
        # convert boxes_h, boxes_o, to torch.tensor
        boxes_h = torch.Tensor(boxes_h.asnumpy()).to(device)
        boxes_o = torch.Tensor(boxes_o.asnumpy()).to(device)

        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)
        
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        
        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
        
        if self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1  ## target['labels']: verb/action
        else:
            labels[x, targets['hoi'][y]] = 1
        # print("#(labels==1) = ", torch.sum(labels))
        
        labels = ms.Tensor(labels.cpu().numpy())
        return labels

    def compute_interaction_loss_ms(self, boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats,):
        ## bh, bo: indices of boxes
        labels = P.Concat()([self.associate_with_ground_truth_ms(bx[h], bx[o], target) for (bx, h, o, target) in zip(boxes, bh, bo, targets)])
        
        prior = P.Concat(1)(prior).prod(0)
        x, y = ms.ops.nonzero(prior).unbind(1)
        num_one_label = ms.ops.sum(labels)
        logits = P.Concat()(logits) 
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]
        
        n_p = len(ms.ops.nonzero(labels))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            # n_p_distll = torch.as_tensor([n_p_distll], device='cuda')
            dist.barrier() 
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

            # dist.all_reduce(n_p_distll)
            # n_p_distll = (n_p_distll / world_size).item()
            # n_p = (n_p.true_divide(world_size)).item()
        
        loss = binary_focal_loss_with_logits_ms(
            P.Log()(prior / (1 + P.Exp()(-logits) - prior) + 1e-08), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
            )
        
        return loss / n_p

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats,): ### loss
        ## bx, bo: indices of boxes
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])

        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        num_one_label = torch.sum(labels)
        logits = torch.cat(logits) 
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]
        
        n_p = len(torch.nonzero(labels))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            # n_p_distll = torch.as_tensor([n_p_distll], device='cuda')
            dist.barrier() 
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

            # dist.all_reduce(n_p_distll)
            # n_p_distll = (n_p_distll / world_size).item()
            # n_p = (n_p.true_divide(world_size)).item()
        
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
            )
        
        return loss / n_p

    def prepare_region_proposals(self, results): ## √ detr extracts the human-object pairs
        region_props = []
        for res in results:
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                # keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                ## use topk instead of argsort
                keep_h = sc[hum].topk(min(self.min_instances, len(sc[hum])), largest=True)[1]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                # keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = sc[hum].topk(min(self.max_instances, len(sc[hum])), largest=True)[1]             
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                # keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = sc[obj].topk(min(self.min_instances, len(sc[obj])), largest=True)[1]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                # keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = sc[obj].topk(min(self.max_instances, len(sc[obj])), largest=True)[1]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
            ))

        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes): ### √
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size,  in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections

    def get_prior(self, region_props, image_size, prior_method): ##  for adapter module training
        
        max_feat = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
        priors = torch.zeros((len(region_props),max_length, max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] / scale_fct[b_idx][None,:]
            scores = props['scores']
            labels = props['labels']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h,n)
                # sys.exit()
            
            object_embs = self.object_embedding[labels]
            mask[b_idx,:n] = False
            
            if self.prior_type == 'cbe':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
                priors[b_idx,:n,5:self.visual_output_dim+5] = object_embs
                # priors[b_idx,:n,512+5:] = unary_tokens
            elif self.prior_type == 'cb':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            elif self.prior_type == 'ce':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
                priors[b_idx,:n,1:self.visual_output_dim+1] = object_embs
            elif self.prior_type == 'be':
                priors[b_idx,:n,:4] = boxes
                priors[b_idx,:n,4:self.visual_output_dim+4] = object_embs
            elif self.prior_type == 'c':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
            elif self.prior_type == 'b':
                priors[b_idx,:n,:4] = boxes
            elif self.prior_type == 'e':
                priors[b_idx,:n,:self.visual_output_dim] = object_embs
            # else:
            #     raise NotImplementedError

        if prior_method == 0:
            priors = self.priors_downproj(priors)
        elif prior_method == 1:
            pair_wise_priors = []
            for b_idx, props in enumerate(region_props):
                boxes = props['boxes'] / scale_fct[b_idx][None,:]
                scores = props['scores']
                labels = props['labels']
                is_human = labels == self.human_idx
                n_h = torch.sum(is_human); n = len(boxes)
                if n_h == 0 or n <= 1:
                    pair_wise_priors.append(torch.zeros(0, 0), )
                    print(n_h,n)
                    continue
                instance_wise_prior = priors[b_idx, :n]
                # Get the pairwise indices
                x, y = torch.meshgrid(
                    torch.arange(n, device=instance_wise_prior.device),
                    torch.arange(n, device=instance_wise_prior.device)
                )
                # Valid human-object pairs
                x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
                if len(x_keep) == 0:
                    # Should never happen, just to be safe
                    return 0
                    # raise ValueError("There are no valid human-object pairs")
                
                # extract single roi features
                sub_prior = instance_wise_prior[x_keep]
                obj_prior = instance_wise_prior[y_keep]
                
                pair_wise_priors.append(torch.cat((sub_prior, obj_prior), dim=-1))
            
            max_length = max(p.shape[0] for p in pair_wise_priors)
            mask = torch.ones((len(region_props),max_length),device=region_props[0]['boxes'].device)
            priors = torch.zeros((len(region_props),max_length, max_feat*2), dtype=torch.float32, device=region_props[0]['boxes'].device)
            for b_idx, props in enumerate(region_props):
                num_pair = pair_wise_priors[b_idx].shape[0]
                if num_pair > 0:
                    mask[b_idx, :num_pair] = False
                    priors[b_idx, :num_pair] = pair_wise_priors[b_idx]
            priors = self.priors_downproj(priors)   
        elif prior_method == 2:
            priors = self.learnable_prior.unsqueeze(0).repeat(len(region_props), 1, 1)
            mask = torch.zeros((priors.shape[0], priors.shape[1]),device=region_props[0]['boxes'].device)

        return (priors, mask)    
    
    def get_prior_ms(self, region_props, image_size, prior_method):
        max_feat = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = ms.ops.ones((len(region_props),max_length), dtype=ms.bool_)
        priors = ms.ops.zeros((len(region_props),max_length, max_feat), dtype=ms.float32)
        img_h, img_w = image_size.unbind(-1)
        scale_fct = ms.ops.stack([img_w, img_h, img_w, img_h], axis=1)

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] / scale_fct[b_idx][None,:]
            scores = props['scores']
            labels = props['labels']
            is_human = labels == self.human_idx
            n_h = ms.ops.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h,n)
                # sys.exit()
            
            object_embs = self.object_embedding[labels]
            mask[b_idx,:n] = False

            assert self.prior_type == 'cbe', "prior_type != cbe"
            priors[b_idx,:n,:5] = ms.ops.cat((scores.unsqueeze(-1),boxes),axis=-1)
            priors[b_idx,:n,5:self.visual_output_dim+5] = object_embs

        assert self.prior_method == 0, "prior_method != 0"
        priors = self.priors_downproj(priors)

        return (priors, mask)

    def forward(self,
        images: List[torch.Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        batch_size = len(images)
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]
        device = images_clip[0].device
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images_clip
        ], device=device)
        image_sizes_orig = torch.as_tensor([
            im.size()[-2:] for im in images_orig
            ], device=device)
        
        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)
        features, pos = self.detector.backbone(images_orig)
        src, mask = features[-1].decompose()
        # assert mask is not None2
        hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])
        outputs_class = self.detector.class_embed(hs) # 6x8x100x81 or 6x8x100x92
        outputs_coord = self.detector.bbox_embed(hs).sigmoid() # 6x8x100x4 
        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'
        
        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)

        region_props = self.prepare_region_proposals(results) # List(dict{}), dict_keys(['boxes', 'scores', 'labels']), values: torch.tensor
        
        if self.training and self.ms:
            ms.set_context(device_target='GPU')
            ## convert region_props to ms.tensor
            for i in range(len(region_props)):
                for k, v in region_props[i].items():
                    region_props[i][k] = ms.Tensor(v.cpu().numpy())
            ## convert image_sizes to ms.tensor
            image_sizes = ms.Tensor(image_sizes.cpu().numpy())
            prior, prior_mask = self.get_prior_ms(region_props, image_sizes, self.prior_method)

            images_clip = nested_tensor_from_tensor_list(images_clip)
            images_clip = images_clip.decompose()[0]
            images_clip = ms.Tensor(images_clip.cpu().numpy())
            feat_global, feat_local = self.clip_head.image_encoder(images_clip, prior, prior_mask)

            logits, prior, bh, bo, objects, gt_feats, pair_feats = self.compute_roi_embeddings_ms(feat_local, image_sizes, region_props)
            boxes = [r['boxes'] for r in region_props] 
            
            if self.training:
                interaction_loss = self.compute_interaction_loss_ms(boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats)
                loss_dict = dict(
                    interaction_loss=interaction_loss
                )
                return interaction_loss, logits

        prior, prior_mask = self.get_prior(region_props, image_sizes, self.prior_method) ## priors: (prior_feat, mask): (batch_size*14*64, batch_size*14)

        images_clip = nested_tensor_from_tensor_list(images_clip)
        images_clip = images_clip.decompose()[0]
        if self.ms:
            ms.set_context(device_target='GPU')
            images_clip, prior, prior_mask = ms.Tensor(images_clip.cpu().numpy()), ms.Tensor(prior.cpu().numpy()), ms.Tensor(prior_mask.cpu().numpy())
        feat_global, feat_local = self.clip_head.image_encoder(images_clip, prior, prior_mask)
        if self.ms:
            feat_global, feat_local = feat_global.asnumpy(), feat_local.asnumpy()
            feat_global, feat_local = torch.tensor(feat_global).to(device), torch.tensor(feat_local).to(device)

        logits, prior, bh, bo, objects, gt_feats, pair_feats = self.compute_roi_embeddings(feat_local, image_sizes, region_props)
        boxes = [r['boxes'] for r in region_props] 
        
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats)
            loss_dict = dict(
                interaction_loss=interaction_loss
            )
            return loss_dict
        if len(logits) == 0:
            print(targets)
            return None
        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections


def get_multi_prompts(classnames):   ## https://github.com/openai/CLIP/blob/main/data/prompts.md, 
    templates = ['a photo of a person {}.',
                'a video of a person {}.',
                'a example of a person {}.',
                'a demonstration of a person {}.',
                'a photo of the person {}.',
                'a video of the person {}.',
                'a example of the person {}.', 
                'a demonstration of the person {}.',
                
                # 'a photo of a person during {}.',
                # 'a video of a person during {}.',
                # 'a example of a person during {}.',
                # 'a demonstration of a person during {}.',
                # 'a photo of the person during {}.',
                # 'a video of the person during {}.',
                # 'a example of the person during {}.',
                # 'a demonstration of the person during {}.',

                # 'a photo of a person performing {}.',
                # 'a video of a person performing {}.',
                # 'a example of a person performing {}.',
                # 'a demonstration of a person performing {}.',
                # 'a photo of the person performing {}.',
                # 'a video of the person performing {}.',
                # 'a example of the person performing {}.',
                # 'a demonstration of the person performing {}.',
                
                # 'a photo of a person practicing {}.',
                # 'a video of a person practicing {}.',
                # 'a example of a person practicing {}.',
                # 'a demonstration of a person practicing {}.',
                # 'a photo of the person practicing {}.',
                # 'a video of the person practicing {}.',
                # 'a example of the person practicing {}.',
                # 'a demonstration of the person practicing {}.',
                ]
    hico_texts = [' '.join(name.split(' ')[5:]) for name in classnames]
    all_texts_input = []
    for temp in templates:
        texts_input = torch.cat([clip.tokenize(temp.format(text)) for text in hico_texts ])
        all_texts_input.append(texts_input)
    all_texts_input = torch.stack(all_texts_input,dim=0)
    return all_texts_input

@torch.no_grad()
def get_origin_text_emb(args, clip_model, tgt_class_names, obj_class_names):
    use_templates = args.use_templates
    if use_templates == False:
        text_inputs = torch.cat([clip.tokenize(classname) for classname in tgt_class_names])
    elif use_templates:
        text_inputs = get_multi_prompts(tgt_class_names)
        bs_t, nums, c = text_inputs.shape
        text_inputs = text_inputs.view(-1, c)

    ## convert text_inputs to mindspore tensor from torch tensor
    if args.ms:
        text_inputs = ms.Tensor(text_inputs.numpy())
    with torch.no_grad():
        origin_text_embedding = clip_model.encode_text(text_inputs)
        origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(dim=-1, keepdim=True) # text embeddings of hoi 117*512 or 600*512

    obj_text_inputs = torch.cat([clip.tokenize(obj_text) for obj_text in obj_class_names])
    if args.ms:
        obj_text_inputs = ms.Tensor(obj_text_inputs.numpy())
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
    
    if args.ms: ## convert to torch tensor from mindspore tensor
        origin_text_embedding = torch.tensor(origin_text_embedding.asnumpy())
        object_embedding = torch.tensor(object_embedding.asnumpy())

    return origin_text_embedding, object_embedding


def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path, num_anno, verb2interaction=None):
    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model']) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    
    if args.ms:
        from mindspore import load_checkpoint
        clip_state_dict = load_checkpoint(clip_model_path)
    else:
        clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()

    if args.ms:
        from output import CLIP_models_adapter_prior2
    else:
        import CLIP_models_adapter_prior2

    clip_model = CLIP_models_adapter_prior2.build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, adapter_pos=args.adapter_pos, adapter_num_layers=args.adapter_num_layers)
    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())

    model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)

    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]

    origin_text_embeddings, object_embedding = get_origin_text_emb(args, clip_model=clip_model, tgt_class_names=classnames, obj_class_names=obj_class_names)
    origin_text_embeddings = origin_text_embeddings.clone().detach()
    object_embedding = object_embedding.clone().detach()
    
    detector = UPT(args,
        detr, postprocessors['bbox'], model, origin_text_embeddings, object_embedding,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        num_anno = num_anno,
        # verb2interaction = verb2interaction,
        use_mlp_proj = args.use_mlp_proj,
    )
    return detector

