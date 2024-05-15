from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common.initializer import initializer, XavierUniform

from model.layers import Conv1D, mask_logits


class CQAttention(nn.Cell):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        self.w4C = mindspore.Parameter(initializer('xavier_uniform', (dim, 1)), requires_grad=True)
        self.w4Q = mindspore.Parameter(initializer('xavier_uniform', (dim, 1)), requires_grad=True)
        self.w4mlu = mindspore.Parameter(initializer('xavier_uniform', (1, 1, dim)), requires_grad=True)

        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def construct(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(axis=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(axis=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.swapaxes(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = P.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = P.matmul(P.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = P.cat([context, c2q, P.mul(context, c2q), P.mul(context, q2c)], axis=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = P.matmul(context, self.w4C).broadcast_to((-1, -1, q_seq_len))  # (batch_size, c_seq_len, q_seq_len)
        subres1 = P.matmul(query, self.w4Q).swapaxes(1, 2).broadcast_to((-1, c_seq_len, -1))
        subres2 = P.matmul(context * self.w4mlu, query.swapaxes(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class WeightedPool(nn.Cell):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        self.weight = mindspore.Parameter(initializer('xavier_uniform', (dim, 1)), requires_grad=True)

    def construct(self, x, mask):
        alpha = mindspore.numpy.tensordot(x, self.weight, axes=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(axis=1)(alpha)
        pooled_x = P.matmul(x.swapaxes(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class CQConcatenate(nn.Cell):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def construct(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).tile((1, c_seq_len, 1))  # (batch_size, c_seq_len, dim)
        output = P.cat([context, pooled_query], axis=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


class VSLFuser(nn.Cell):

    def __init__(self, dim=128, drop_rate=0., **kwargs):
        super().__init__()
        self.cq_attention = CQAttention(dim=dim, drop_rate=drop_rate)
        self.cq_concat = CQConcatenate(dim=dim)

    def construct(self, vfeats=None, qfeats=None, vmask=None, qmask=None, **kwargs):
        assert None not in [vfeats, qfeats, vmask, qmask]
        feats = self.cq_attention(vfeats, qfeats, vmask, qmask)
        feats = self.cq_concat(feats, qfeats, qmask)
        return P.relu(feats)
