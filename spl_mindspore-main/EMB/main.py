import os

import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F

import numpy as np
from tqdm import tqdm

from model.emb import build_optimizer_and_scheduler
from model.emb import EMB
from model.threshold import SigmoidThreshold
from util.data_util import load_video_features, save_json, load_json, time_to_index
from util.data_gen import gen_or_load_dataset
from util.data_loader import get_train_loader, get_test_loader
from util.runner_utils import set_th_config, convert_length_to_mask, eval_test, \
    ProgressMeters, AverageMeters, batch_index_to_time, calculate_batch_iou, \
    get_logger, save_checkpoint, load_checkpoint
from util.args_parser import gconfigs, parser


clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


def setup(configs):
    # set tensorflow configs
    set_th_config(configs.seed)

    # prepare or load dataset
    dataset = gen_or_load_dataset(configs)
    configs.char_size = dataset['n_chars']
    configs.word_size = dataset['n_words']

    # get train and test loader
    visual_features = load_video_features(os.path.join('data', 'features', configs.task, configs.fv), configs.max_pos_len)
    train_loader = get_train_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs)
    val_loader = None if dataset['val_set'] is None else get_test_loader(dataset['val_set'], visual_features, configs)
    test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs)
    configs.num_train_steps = len(train_loader) * configs.epochs
    num_train_batches = len(train_loader)
    num_val_batches = 0 if val_loader is None else len(val_loader)
    num_test_batches = len(test_loader)

    # Device configuration
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target='GPU')

    # create model dir
    model_dir = os.path.join(configs.model_dir, configs.task, configs.model_name)
    # save snapshot
    if configs.deploy: os.makedirs(model_dir, exist_ok=True)
    # get logger, log to file if deploy
    logger = get_logger(level=('DEBUG' if not configs.deploy else 'INFO'),
                        to_file=(None if not configs.deploy else os.path.join(model_dir, 'log')))

    return dataset, train_loader, val_loader, test_loader, num_train_batches, num_val_batches, num_test_batches, model_dir, logger


def generate_h_labels(s_idx, e_idx, vfeat_len):
    extend = 0.1
    h_labels = np.zeros(vfeat_len, dtype=np.int64)
    extend_len = round(extend * float(e_idx - s_idx + 1))
    if extend_len > 0:
        st_ = max(0, s_idx - extend_len)
        et_ = min(e_idx + extend_len, vfeat_len - 1)
        h_labels[int(st_):int(et_ + 1)] = 1
    else:
        h_labels[int(s_idx):int(e_idx + 1)] = 1
    return h_labels


class Trainer:
    def __init__(self, net, optimizer, train_dataset, eval_dataset, threshold, logger, configs):
        self.net = net
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.weights = self.opt.parameters
        self.records = None
        self.threshold = threshold
        self.configs = configs
        self.logger = logger
        self.epoch = 0
        self.hyper_map = P.HyperMap()

        self.value_and_grad = mindspore.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)

    def forward_fn(self, vfeats, vfeat_lens, word_ids, s_labels, e_labels, h_labels, durations, video_mask, query_mask, moments):
        # compute logits
        h_score, start_logits, end_logits, proposals, p_score, p_mask = self.net(word_ids, vfeats, video_mask, query_mask)
        p_time = batch_index_to_time(proposals[...,0], proposals[...,1], 
            vfeat_lens.unsqueeze(-1), durations.unsqueeze(-1))
        p_time = P.stack(p_time, axis=-1)
        p_time = p_time * p_mask.unsqueeze(-1)
        # compute loss
        rank_loss, sample_weight, soft_labels = self.net.compute_ranking_loss(moments, p_time, p_score, 
                    p_mask, min_iou=self.configs.min_iou, max_iou=self.configs.max_iou, noise_alpha=self.configs.noise_alpha, noise_beta=self.configs.noise_beta, use_reweight=self.epoch > self.configs.epochs // 2)
        highlight_loss = self.net.compute_highlight_loss(h_score, h_labels, video_mask,
            P.stack([s_labels, e_labels], axis=-1), proposals, p_score, p_mask,
            threshold=self.threshold(self.epoch, reverse=True), extend=self.configs.extend, sample_weight=sample_weight)
        loc_loss = self.net.compute_loss(start_logits, end_logits, s_labels, 
            e_labels, proposals, p_score, p_mask,
            threshold=self.threshold(self.epoch, reverse=True), sample_weight=sample_weight)
        total_loss = (loc_loss + 
                        self.configs.highlight_lambda * highlight_loss +
                        self.configs.rank_lambda * rank_loss)

        return total_loss, loc_loss, highlight_loss, rank_loss, soft_labels

    def train_single(self, data, epoch):
        indexes, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels = data
        records = self.train_dataset.children[0].source.get_record(indexes.asnumpy().tolist())
        # soft label
        if self.configs.use_refine and epoch > self.configs.epochs // 2:
            for idx in range(len(records)):
                if records[idx]['sample_id'] not in self.soft_labels_last:
                    continue
                s_labels[idx] = self.soft_labels_last[records[idx]['sample_id']][0]
                e_labels[idx] = self.soft_labels_last[records[idx]['sample_id']][1]
                h_labels[idx][:vfeat_lens[idx]] = self.soft_labels_last[records[idx]['sample_id']][2]
        # prepare meta
        durations = [record['duration'] for record in records]
        durations = mindspore.tensor(durations, dtype=mindspore.float32)
        # generate mask
        query_mask = (P.zeros_like(word_ids) != word_ids).float()
        video_mask = convert_length_to_mask(vfeat_lens, self.configs.max_pos_len)
        moments = batch_index_to_time(s_labels, e_labels, vfeat_lens, durations)
        moments = P.stack(moments, axis=-1)

        (total_loss, loc_loss, highlight_loss, rank_loss, soft_labels), grads = self.value_and_grad(
                    vfeats, vfeat_lens, word_ids, s_labels, e_labels, h_labels, durations, video_mask, query_mask, moments)
        
        if self.configs.use_refine:
            for idx in range(len(soft_labels)):
                s_ind, e_ind, _ = time_to_index(soft_labels[idx][0], soft_labels[idx][1], float(vfeat_lens[idx]), records[idx]['duration'])
                self.soft_labels_last[records[idx]['sample_id']] = [s_ind, e_ind, generate_h_labels(s_ind, e_ind, float(vfeat_lens[idx]))]

        # compute and apply gradients
        grads = self.hyper_map(F.partial(clip_grad, 1, self.configs.clip_norm), grads)
        self.opt(grads)

        return total_loss, loc_loss, highlight_loss, rank_loss


    def train_and_eval(self, epochs, model_dir):
        # start training
        num_train_batches = len(self.train_dataset)
        log_period = num_train_batches // self.configs.period
        # eval_period = num_train_batches // 2
        eval_period = num_train_batches
        best_results = (-1., -1., -1., -1)
        self.logger.debug('Start training...')
        global_step = 0
        self.soft_labels_last = {}
        for epoch in tqdm(range(epochs), desc="Overall", leave=True):
            self.epoch = epoch
            meters = ProgressMeters(len(self.train_dataset), AverageMeters(), 
                                    prefix="Epoch: [%3d]" % (epoch + 1))
            self.net.set_train(True)

            iterator = self.train_dataset.create_tuple_iterator(num_epochs=1, do_copy=False)
            for local_step, data in enumerate(tqdm(iterator, total=num_train_batches, 
                                            desc='Epoch %3d' % (epoch + 1), leave=False)):
                global_step += 1
                
                total_loss, loc_loss, highlight_loss, rank_loss = self.train_single(data, epoch)
                # meters.update(Threshold=threshold(epoch, reverse=True), 
                meters.update(High=highlight_loss, Rank=rank_loss, Loc=loc_loss, 
                            Loss=total_loss)
                if local_step % log_period == 0:
                    meters.display(local_step, self.logger)
                # evaluate
                # if global_step % eval_period == 0 or global_step % num_train_batches == 0:
                #     self.net.set_train(False)

                #     r1i3, r1i5, r1i7, mi, score_str = eval_test(model=self.net, data_loader=self.eval_dataset,
                #         mode='test', epoch=epoch + 1, global_step=global_step, elastic=self.configs.elastic, max_len=self.configs.max_pos_len)
                #     self.logger.info('Epoch: %3d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % (
                #         epoch + 1, global_step, r1i3, r1i5, r1i7, mi))
                #     results = (r1i3, r1i5, r1i7, mi)
                #     if self.configs.deploy:
                #         save_checkpoint(model_dir, self.net, is_best=(sum(results) > sum(best_results)))
                #     if sum(results) > sum(best_results): best_results = results
                #     self.net.set_train(True)
        if self.configs.deploy:
            save_checkpoint(model_dir, self.net, filename='checkpoint_%d.ckpt'%epoch)
        self.logger.debug('Done training')
        # self.logger.info('Best results yielded - r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % best_results)
        return best_results

# training
def train(configs):
    (dataset, train_loader, val_loader, test_loader, num_train_batches, 
     num_val_batches, num_test_batches, model_dir, logger) = setup(configs)
    if configs.deploy:
        save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    # build model
    model = EMB(configs=configs, word_vectors=dataset['word_vector'])
    optimizer = build_optimizer_and_scheduler(model, configs=configs)
    threshold = SigmoidThreshold(start=0, end=configs.epochs, low=configs.threshold)
    
    trainer = Trainer(model, optimizer, train_loader, test_loader, threshold, logger, configs)
    best_results = trainer.train_and_eval(configs.epochs, model_dir)

    return best_results


def test(configs):
    # setup
    (dataset, train_loader, val_loader, test_loader, num_train_batches, 
     num_val_batches, num_test_batches, model_dir, logger) = setup(configs)
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    # load previous configs
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # build model
    model = EMB(configs=configs, word_vectors=dataset['word_vector'])
    # get last checkpoint file
    filename = load_checkpoint(model_dir)
    mindspore.load_checkpoint(filename, model)
    model.set_train(False)
    r1i3, r1i5, r1i7, mi, _ = eval_test(model=model, data_loader=test_loader, 
        mode='test', elastic=configs.elastic, max_len=configs.max_pos_len)
    logger.info("\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m")
    logger.info("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m")
    logger.info("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m")
    logger.info("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m")
    return r1i3, r1i5, r1i7, mi


if __name__ == '__main__':
    if gconfigs.mode.lower() == 'train':
        train(gconfigs)
    elif gconfigs.mode.lower() == 'test':
        test(gconfigs)
    else:
        raise NotImplementedError('mode should be one of ("train", "test")')
