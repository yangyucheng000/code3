import copy
import math
import collections

import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common.initializer import initializer, XavierUniform

from util.runner_utils import calculate_batch_iou


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.float()
    return inputs + (1.0 - mask) * mask_value


def mask_pooling(ntensor, nmask, mode='avg', dim=-1):
    """mark pooling over the given tensor
    """
    if mode in ['avg', 'sum']:
        ntensor = (ntensor * nmask).sum(dim)
        if mode == 'avg':
            ntensor /= nmask.sum(dim).clamp(min=1e-10)
    elif mode == 'max':
        ntensor = mask_logits(ntensor, nmask)
        ntensor = ntensor.max(dim)
    else:
        raise NotImplementedError('Only sum/average/max pooling have been implemented')
    return ntensor


def downscale1d(x, scale=1, dim=None, mode='max'):
    # pad at tail
    padding = (scale - x.shape[-1] % scale) % scale
    x = P.pad(x, (0, padding), value=(0 if mode=='sum' else -1e-30))
    # downscale
    if mode == 'max':
        mpool1 = nn.MaxPool1d(kernel_size=scale, stride=scale)
        return mpool1(x)
    raise NotImplementedError('downscale1d implemented only "max" and "sum" mode')


class Conv1D(nn.Cell):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, has_bias=bias, pad_mode='pad')

    def construct(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.swapaxes(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.swapaxes(1, 2)  # (batch_size, seq_len, dim)


class DepthwiseSeparableConvBlock(nn.Cell):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.conv_layers = nn.CellList([
            nn.SequentialCell([
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, has_bias=False, pad_mode='pad'),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, has_bias=True, pad_mode='pad'),
                nn.ReLU(),
            ]) for _ in range(num_layers)])
        self.layer_norms = nn.CellList([nn.LayerNorm(normalized_shape=dim, epsilon=1e-06) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=drop_rate)

    def construct(self, x):
        output = x  # (batch_size, seq_len, dim)
        for idx, conv_layer in enumerate(self.conv_layers):
            residual = output
            output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
            output = output.swapaxes(1, 2)  # (batch_size, dim, seq_len)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.swapaxes(1, 2) + residual  # (batch_size, seq_len, dim)
        return output


class WordEmbedding(nn.Cell):
    def __init__(self, num_words, word_dim, drop_rate, to_norm=False, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = mindspore.Parameter(P.zeros(size=(1, word_dim), dtype=mindspore.float32), requires_grad=False)
            self.unk_vec = mindspore.Parameter(initializer('xavier_uniform', (1, word_dim), mindspore.float32), requires_grad=True)
            self.glove_vec = mindspore.Parameter(mindspore.tensor(word_vectors, dtype=mindspore.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)
        self.to_norm = to_norm

    def construct(self, word_ids):
        if self.is_pretrained:
            # word_emb = P.EmbeddingLookup()(P.cat([self.pad_vec, self.unk_vec, self.glove_vec], axis=0), word_ids, 0)
            # word_emb = F.embedding(word_ids, P.Concat(0)([self.pad_vec, self.unk_vec, self.glove_vec]),
            #                        padding_idx=0)
            word_emb = P.gather(P.cat([self.pad_vec, self.unk_vec, self.glove_vec], axis=0), word_ids, 0)
        else:
            word_emb = self.word_emb(word_ids)
        if self.to_norm:
            word_emb = P.L2Normalize(-1, 1e-12)(word_emb)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Cell):
    def __init__(self, num_chars, char_dim, drop_rate, to_norm=False):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.CellList([
            nn.SequentialCell([
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), pad_mode='pad', padding=0, has_bias=True),
                nn.ReLU()
            ]) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=drop_rate)
        self.to_norm = to_norm

    def construct(self, char_ids):
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, c_seq_len, char_dim)
        if self.to_norm:
            char_emb = P.L2Normalize(-1, 1e-12)(char_emb)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)  # (batch_size, char_dim, w_seq_len, c_seq_len)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = P.max(output, axis=3, keepdims=False)  # reduce max (batch_size, channel, w_seq_len)
            char_outputs.append(output)
        char_output = P.cat(char_outputs, axis=1)  # (batch_size, sum(channels), w_seq_len)
        return char_output.permute(0, 2, 1)  # (batch_size, w_seq_len, sum(channels))


class Embedding(nn.Cell):
    def __init__(self, num_words, num_chars, word_dim, char_dim, drop_rate, 
                    out_dim, word_vectors=None, to_norm=False):
        super(Embedding, self).__init__()
        self.word_emb = WordEmbedding(num_words, word_dim, drop_rate, to_norm, word_vectors=word_vectors)
        # self.char_emb = CharacterEmbedding(num_chars, char_dim, drop_rate, to_norm)
        # output linear layer
        in_dim = word_dim
        self.linear = Conv1D(in_dim=in_dim, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def construct(self, word_ids):
        word_emb = self.word_emb(word_ids)  # (batch_size, w_seq_len, word_dim)
        emb = self.linear(word_emb)  # (batch_size, w_seq_len, dim)
        return emb


class VisualProjection(nn.Cell):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)

    def construct(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        return output


class HighLightLayer(nn.Cell):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def construct(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12, sample_weight=None):
        labels = labels.astype(mindspore.float32)
        weights = P.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.astype(mindspore.float32)
        if sample_weight is not None:
            # loss_scale = loss_per_location.sum().data / ((loss_per_location * sample_weight).sum().data + 1e-9)
            loss_scale = loss_per_location.sum() / ((loss_per_location * sample_weight).sum() + 1e-9)
            loss_scale = P.stop_gradient(loss_scale)
            loss_per_location = loss_per_location * sample_weight * loss_scale
        loss = P.sum(loss_per_location * mask) / (P.sum(mask) + epsilon)
        return loss


class ConditionedPredictor(nn.Cell):
    def __init__(self, dim, predictor):
        super(ConditionedPredictor, self).__init__()
        self.start_encoder = predictor
        self.end_encoder = copy.deepcopy(self.start_encoder)
        self.start_block = nn.SequentialCell([
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        ])
        self.end_block = nn.SequentialCell([
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        ])

    def construct(self, x, mask):
        # encode
        start_features = self.start_encoder(target=x, tmask=mask)  # (batch_size, seq_len, dim)
        end_features = self.end_encoder(target=start_features, tmask=mask)
        # predict
        start_logits = self.start_block(P.cat([start_features, x], axis=2))  # (batch_size, seq_len, 1)
        end_logits = self.end_block(P.cat([end_features, x], axis=2))
        start_logits = mask_logits(start_logits.squeeze(2), mask=mask)
        end_logits = mask_logits(end_logits.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(axis=1)(start_logits)
        end_prob = nn.Softmax(axis=1)(end_logits)
        outer = P.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = P.triu(outer, diagonal=0)
        _, start_index = P.max(P.max(outer, axis=2)[0], axis=1)  # (batch_size, )
        _, end_index = P.max(P.max(outer, axis=1)[0], axis=1)  # (batch_size, )
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return start_loss + end_loss
