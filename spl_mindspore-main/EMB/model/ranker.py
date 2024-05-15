from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as P

from model.layers import mask_logits
from util.runner_utils import calculate_batch_iou

all_soft_labels = []
static_conv_kernel = P.ones((1, 1, 3, 3))

class Conv2dRanker(nn.Cell):

    def __init__(self, dim=128, kernel_size=3, num_layers=4, **kwargs):
        super().__init__()
        self.kernel = kernel_size
        self.encoders = nn.CellList([
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, pad_mode='pad', padding=kernel_size // 2, has_bias=True) 
        for _ in range(num_layers)])
        self.predictor = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1, pad_mode='pad', has_bias=True)

    @staticmethod
    def get_padded_mask_and_weight(mask, conv):
        masked_weight = P.round(P.conv2d(mask.float(), 
            static_conv_kernel,
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation, pad_mode='pad'))
        masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0] #conv.kernel_size[0] * conv.kernel_size[1]
        padded_mask = masked_weight > 0
        return padded_mask, masked_weight

    def construct(self, x, mask):
        # convert to 2d if input is flat
        if x.dim() < 4:
            B, N2, D = x.shape
            assert int(math.sqrt(N2)) == math.sqrt(N2)
            N = int(math.sqrt(N2))
            x2d, mask2d = x.reshape(B, N, N, D), mask.reshape(B, N, N)
        else:
            x2d, mask2d = x, mask
        # x: (<bsz>, <num>, <num>, <dim>) -> (<bsz>, <dim>, <num>, <num>)
        x2d, mask2d = x2d.permute(0, 3, 1, 2), mask2d.unsqueeze(1)
        for encoder in self.encoders:
            # mask invalid features to 0
            x2d = P.relu(encoder(x2d * mask2d))
            _, weights = self.get_padded_mask_and_weight(mask2d, encoder)
            x2d = x2d * P.stop_gradient(weights)
        # preds: (<bsz>, <num>, <num>)
        preds = self.predictor(x2d).view_as(mask)
        preds = mask_logits(preds, mask)
        return preds.sigmoid()

    @staticmethod
    def topk_confident(bounds, scores, mask, moments=None, threshold=0.5, k=1):
        if moments is not None:
            # compute the overlaps between proposals and ground-truth
            overlaps = calculate_batch_iou(bounds, moments.unsqueeze(1))
            # set the scores of proposals with 
            # insufficient overlaps with the ground-truth to -inf
            is_cand = (overlaps >= threshold) * mask
        else:
            is_cand = mask
        masked_scores = mask_logits(scores, is_cand)
        # get topk proposals
        cands_idx = masked_scores.topk(k, dim=1)[1]
        cands = bounds.gather_elements(1, cands_idx.unsqueeze(-1).tile((1, 1, 2)))
        if moments is not None:
            # in training
            # use the ground-truth moment if there is no proposal whose overlaps 
            # with the ground-truth is greater than the threshold
            # for example, when the threshold equal to 1
            has_cand = (is_cand.sum(-1) > 0).view(-1, 1, 1)
            cands = cands * has_cand + moments.unsqueeze(1) * (~has_cand)
        return cands

    @staticmethod
    def compute_loss(moments, bounds, scores, mask, min_iou=0.3, max_iou=0.7, noise_alpha=0.75, noise_beta=0.75, use_reweight=False):
        # assert scores.shape == mask.shape and min_iou <= max_iou
        ious = calculate_batch_iou(bounds, moments.unsqueeze(1))

        if min_iou == max_iou:
            targets = (ious >= min_iou).float()
        else:
            targets = (ious - min_iou) / (max_iou - min_iou)
            targets = targets.clamp(min=0, max=1)
        loss = P.binary_cross_entropy(scores, P.stop_gradient(targets), reduction='none') * mask
        
        if use_reweight:
            loc_acc_r = 1 / (1 - ious + 1e-9) * mask
            loc_acc_r = loc_acc_r / (P.sum(loc_acc_r, dim=-1, keepdim=True) / P.sum(mask, dim=-1, keepdim=True))
            cls_conf_r = 1 / (1 - scores + 1e-9) * mask
            cls_conf_r = cls_conf_r / (P.sum(cls_conf_r, dim=-1, keepdim=True) / P.sum(mask, dim=-1, keepdim=True))
            reweight =  noise_alpha * loc_acc_r + (1 - noise_alpha) * cls_conf_r
            tgt_idx = scores.argmax(dim=-1)
            weight = reweight.gather_elements(dim=-1, index=tgt_idx.unsqueeze(-1))
            weight = P.stop_gradient(weight)
            loss_scale = loss.sum() / ((loss * weight).sum() + 1e-9)
            loss = loss * weight * P.stop_gradient(loss_scale)
        else:
            weight = None

        soft_labels = []
        # softscore = (noise_beta * ious + (1-noise_beta) * scores) * mask
        # soft_idx = softscore.argmax(axis=-1)
        # for i, idx in enumerate(soft_idx):
        #     soft_labels.append(bounds[i, int(idx)].asnumpy().tolist())

        return P.sum(loss) / P.sum(mask), weight, soft_labels