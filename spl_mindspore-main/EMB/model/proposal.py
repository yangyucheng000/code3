from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import mindspore
import mindspore.nn as nn
import mindspore.ops as P

from model.layers import mask_logits, downscale1d


class TAN2dProposal(nn.Cell):

    def __init__(self, downscale=8, windows=[16], **kwargs):
        super().__init__()
        # downscale before generating proposals
        self.scale = downscale
        # pooling layers
        self.windows = windows
        layers = []
        for i, window in enumerate(windows):
            layers.extend([nn.MaxPool1d(1, 1) if i == 0 else nn.MaxPool1d(3, 2)])
            layers.extend([nn.MaxPool1d(2, 1) for _ in range(window - 1)])
        self.layers = nn.CellList(layers)
        
        self.mask2d = P.zeros((1, 16, 16))
        self.feat2d = P.zeros((1, 128, 16, 16))
        self.arange_constant = P.arange(0, 16)

    def construct(self, feats=None, mask=None, **kwargs):
        # assert None not in [feats, mask]
        # set all invalid features to a very small values
        # to avoid their impact in maxpooling
        feats = mask_logits(feats, mask.unsqueeze(-1))
        # apply downscale first
        feats = downscale1d(feats.swapaxes(1, 2), scale=self.scale, mode='max')
        scaled_mask = downscale1d(mask.unsqueeze(1), scale=self.scale, mode='max').squeeze(1)
        B, D, N = feats.shape

        mask2d = self.mask2d.broadcast_to((B, N, N))
        feat2d = self.feat2d.broadcast_to((B, D, N, N))
        offset, stride = -1, 1
        for i, window in enumerate(self.windows):
            for j in range(window):
                layer = self.layers[i * len(self.windows) + j]
                if feats.shape[-1] < layer.kernel_size[-1]: break
                offset += stride
                start, end = list(range(0, N - offset, stride)), list(range(offset, N, stride))
                # assume valid features are continual
                mask2d[:, start, end] = scaled_mask[:, end]
                feats = layer(feats)
                feat2d[:,:,start,end] = feats
            stride *= 2
        # mask invalid proposal features to 0
        feat2d = feat2d * mask2d.unsqueeze(1)
        # (B, D, N, N) -> (B, N, N, D)
        feat2d = feat2d.permute(0, 2, 3, 1)
        # generate boundary
        bounds = self.arange_constant
        bounds = bounds.reshape(1, -1).tile((B, 1))
        # (B, N, N, 2)
        bounds = P.stack([bounds.unsqueeze(-1).tile((1, 1, N,)) * self.scale,
            (bounds.unsqueeze(1).tile((1, N, 1,)) + 1) * self.scale - 1], axis=-1)
        # set the largest boundary to the number of items in each sample
        bounds = P.minimum(bounds, mask.sum(-1).reshape(-1, 1, 1, 1).long() - 1)
        # mask invalid proposal
        bounds = bounds * mask2d.unsqueeze(-1).long()
        # make sure for all the valid proposals
        # its endpoint should be greater or equal to its start points
        # assert ((bounds[...,1] >= bounds[...,0]) | ~mask2d.bool()).all()
        # flatten all proposals as output
        feat2d = feat2d.reshape(B, N * N, D)
        bounds = bounds.reshape(B, N * N, 2)
        mask2d = mask2d.reshape(B, N * N)
        return feat2d, P.stop_gradient(bounds), P.stop_gradient(mask2d)
