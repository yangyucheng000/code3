import mindspore
from mindspore import nn, ops,numpy
from loguru import logger
import copy
class BceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = nn.BCELoss(reduction='none')
        self.hinge_loss = False

    def linear_scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d, epoch):
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        loss = 0
        iou1d = self.linear_scale(iou1d).clamp(0, 1)
        loss += self.bceloss(scores1d, iou1d).mean()
        return loss

def box_length(boxes):
    return boxes[:, 1] - boxes[:, 0]

def box_iou(boxes1, boxes2):
    area1 = box_length(boxes1)
    area2 = box_length(boxes2)
    # try:
    #     logger.info(f'max box1 {boxes1[:, None, 0].shape} ')
    # except:
    #     logger.info(f'error max1 {boxes1.shape}，{boxes2.shape}')
    # logger.info(f'max box2 {boxes2[:, 0].shape}')
    max_start = ops.maximum(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
    # logger.info(f'max_start: {max_start.shape}')
    # logger.info(f'min {boxes1.shape} , {boxes2.shape}')
    min_end = ops.minimum(boxes1[:, None, 1].float(), boxes2[:, 1].float()).float()  # [N,M]
    inter = (min_end - max_start).clamp(min=0) 
    # logger.info(f'inter: {inter.shape}，area2:{area2.shape}，area1:{area1.shape}')
    union = area1[:, None] + area2 - inter
    iou = inter / union
    # logger.info(f'iou: {iou.shape}')
    return iou

def build_bce_loss(cfg, mask2d):
    min_iou = cfg.MODEL.TRM.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.TRM.LOSS.MAX_IOU
    return BceLoss(min_iou, max_iou, mask2d)


class PhraseLoss(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.sent_neg_iou = cfg.MODEL.TRM.LOSS.SENT_NEG_IOU
        self.norm = ops.L2Normalize()
    
    def pos_phr_loss(self, score1, score2, margin):
        """
            score1, score2: score maps of phrases in each sentence, (num_phr-1) x num_prop
        """


    def __call__(self, feat2ds, phr_feats, iou2ds, gt_proposals):
        """
            feat2ds: B x C x T x T
            phr_feats: list(B) list(num_sent) num_phr x C
            iou2ds: list(B) num_sent x T x T
            gt_proposals: list(B) num_sent x 2, with format [start, end], unit being seconds (frame/fps)
        """
        B, C, _, _ = feat2ds.size()
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1)
        feat1ds_norm = self.norm(feat1ds, axis=1)

        for i, (phr_feat, iou2d, gt_per_video) in enumerate(zip(phr_feats, iou2ds, gt_proposals)):
            feat1d = feat1ds_norm[i, :, :]  # C x num_prop
            phr_feat = ops.stack(phr_feat)    # num_sent x num_phr x C
            sent_feat = phr_feat[:, 0, :]  # num_sent x C
            phr_feat = phr_feat[:, 1:, :]  # num_sent x (num_phr-1) x C
            iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1) # num_sent x num_prop

            num_sent, num_phr, _ = phr_feat.size()
            phr_feat = phr_feat.reshape(num_sent*num_phr, -1)   # (num_sentx(num_phr-1)) x C
            phr_feat = self.norm(phr_feat, axis=1)     # (num_sentx(num_phr-1)) x C

            phr_score = ops.mm(phr_feat, feat1d).reshape(num_sent, num_phr, -1) # num_sent x (num_phr-1) x num_prop
            phr_feat = phr_feat.reshape(num_sent, num_phr, -1)  # num_sent x (num_phr-1) x C

            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_neg_iou
            sent_neg_mask = iou_mask.float()    # num_sent x num_sent
            pos_loss = 0
            for sen1 in range(num_sent):
                for sen2 in range(sen1+1, num_sent):
                    if iou_map_per_video[sen1, sen2] > self.sent_neg_iou:
                        pos_loss += self.pos_phr_loss(phr_score[sen1, :, :], phr_score[sen2, :, :], phr_feat[sen1, :, :], phr_feat[sen2, :, :])
            return pos_loss



class ContrastiveLoss(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.T_v = cfg.MODEL.TRM.LOSS.TAU_VIDEO
        self.T_s = cfg.MODEL.TRM.LOSS.TAU_SENT
        # self.cri = nn.CrossEntropyLoss()
        self.neg_iou = cfg.MODEL.TRM.LOSS.NEGATIVE_VIDEO_IOU
        self.top_k = cfg.MODEL.TRM.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL
        self.sent_removal_iou = cfg.MODEL.TRM.LOSS.SENT_REMOVAL_IOU
        self.margin = cfg.MODEL.TRM.LOSS.MARGIN
        self.eps = 1e-6
        self.dataset = cfg.DATASETS.NAME
        self.norm1 = ops.L2Normalize(axis=1)

    def __call__(self, feat2ds, sent_feats, iou2ds, phr_feats, phr_weights, gt_proposals):
        """
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
            iou2ds: list(B) num_sent x T x T
            phr_feats: list(B) list(num_sent) num_phr x C
            phr_weights: list(B) num_sent x num_phr
            gt_proposals: list(B) num_sent x 2, with format [start, end], unit being seconds (frame/fps)
        """
        # prepare tensors
        B, C, _, _ = feat2ds.shape
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1) # only ts < te proposals left [B, C, P]
        feat1ds_norm = self.norm1(feat1ds)  # B x C x num_sparse_selected_proposal
        # logger.info(f'sent_feats {[s.shape for s in sent_feats ]}')
        sent_feat_cat = ops.cat(sent_feats, axis=0)  # sum(num_sent) x C, whole batch
        # logger.info(f'sent_feat_cat {sent_feat_cat.shape}')
        sum_num_sent = sent_feat_cat.shape[0]
        sent_feat_cat_norm = self.norm1(sent_feat_cat)  # sum(num_sent) x C, whole batch
        sent_mask = ops.ones((sum_num_sent, sum_num_sent))
        
        all_num_sent = [0]
        curr_num_sent = 0
        for i in range(len(sent_feats)):
            curr_num_sent += sent_feats[i].shape[0]
            all_num_sent.append(curr_num_sent)
        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou  # remove high iou sentence, keep low iou sentence
            sent_mask[all_num_sent[i]:all_num_sent[i+1], all_num_sent[i]:all_num_sent[i+1]] = iou_mask.float()
        # logger.info(f'ops.ones(sum_num_sent): {ops.ones(sum_num_sent).shape}')
        sent_mask += ops.diag(ops.ones(sum_num_sent)).float()  # add the sentence itself to the denominator in the loss
        margin_mask = ops.diag(ops.ones(sum_num_sent)) * self.margin
        vid_pos_list = []
        vid_neg_list = []
        sent_pos_list = []
        sent_neg_list = []
        
    

        for i, (sent_feat, iou2d) in enumerate(zip(sent_feats, iou2ds)):  # each video in the batch
            # select positive samples
            num_sent_this_batch = sent_feat.shape[0]
            
            feat1d = feat1ds_norm[i, :, :]  # C x num_sparse_selected_proposal
            # logger.info(f'{i} feat1ds_norm: {feat1ds_norm[i].shape}')
            sent_feat = self.norm1(sent_feat)   # num_sent x C
            # logger.info(f'sent_feat: {sent_feat.shape}')
            iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.shape[0], -1) # num_sent x num_sparse_selected_proposal
            topk_index = ops.topk(iou1d, self.top_k, dim=-1)[1]   # num_sent x top_k
            selected_feat = feat1d.index_select(axis=1, index=topk_index.reshape(-1)).reshape(C, -1, self.top_k) # C x num_sent x top_k
            selected_feat = selected_feat.permute(1, 2, 0)  # num_sent x top_k x C
            # positive video proposal with pos/neg sentence samples
            vid_pos = ops.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin   # num_sent x top_k, bmm of (num_sent x top_k x C) and (num_sent x C x 1)
            vid_neg = ops.mm(selected_feat.view(-1, C),
                               sent_feat_cat_norm.t()).reshape(-1, self.top_k, sum_num_sent)    # num_sent x topk x sum(num_sent), mm of (num_sent*top_k x C) and (C x sum(num_sent))
            vid_pos_list.append(copy.deepcopy(vid_pos))
            vid_neg_list.append(vid_neg)
            # positive sentence with pos/neg video proposals
            sent_pos_list.append(vid_pos)
            sent_neg_same_video = ops.mm(sent_feat, feat1d)   # num_sent x num_sparse_selected_proposal
            iou_neg_mask = (iou1d < self.neg_iou).float()   # only keep the low iou proposals as negative samples in the same video
            sent_neg_same_video = iou_neg_mask * sent_neg_same_video    # num_sent x num_sparse_selected_proposal
            feat1d_other_video = feat1ds_norm.index_select(axis=0, index=ops.arange(B)[ops.arange(B) != i]) # (B-1) x C x num_sparse_selected_proposal
            # logger.info(f'feat1d_other_video.transpose(1, 0): {feat1d_other_video.swapaxes(1, 0).shape}')
            feat1d_other_video = feat1d_other_video.swapaxes(1, 0).reshape(C, -1)  # C x ((B-1) x num_sparse_selected_proposal)
            # logger.info(f'feat1d_other_video v2: {feat1d_other_video.shape}')
            sent_neg_other_video = ops.mm(sent_feat, feat1d_other_video)  # num_sent x ((B-1) x num_sparse_selected_proposal)
            sent_neg_all = [copy.deepcopy(vid_pos).unsqueeze(2),
                            sent_neg_same_video.unsqueeze(1).tile((1, self.top_k, 1)),
                            sent_neg_other_video.unsqueeze(1).tile((1, self.top_k, 1))]
            sent_neg_list.append(ops.cat(sent_neg_all, axis=2))    # num_sent x topk x (1 + num_same + num_other)
        # import pdb
        # pdb.set_trace()
        vid_pos = (ops.cat(vid_pos_list, axis=0).swapaxes(0, 1)) / self.T_v   # top_k x num_sent
        vid_neg = ops.cat(vid_neg_list, axis=0).permute(1, 0, 2)   # top_k x this_cat_to_be_sum(num_sent) x sum(num_sent)
        vid_neg = (vid_neg - margin_mask) / self.T_v    # top_k x this_cat_to_be_sum(num_sent) (positive) x sum(num_sent) (negative)
        sent_mask += ops.diag(ops.ones(sum_num_sent)).float()
        vid_neg_exp = (ops.exp(vid_neg) * sent_mask).clamp(min=0, max=1)
        loss_vid = -(vid_pos - ops.log(vid_neg_exp.sum(axis=2, keepdims=False))).mean()
        sent_pos = ops.cat(sent_pos_list, axis=0) / self.T_s
        sent_neg = ops.cat(sent_neg_list, axis=0) / self.T_s
        sent_neg_exp = ops.exp(sent_neg)
        loss_sent = -(sent_pos - ops.log(sent_neg_exp.sum(axis=2, keepdims=False) + self.eps)).mean()
        # logger.info('loss pass')
        # raise
        return loss_vid, loss_sent


def build_contrastive_loss(cfg, mask2d):
    return ContrastiveLoss(cfg, mask2d)

if __name__ == '__main__':
    from trm.config import cfg
    mask2d = mindspore.ops.ones((2, 3, 4, 4))
    c_loss = ContrastiveLoss(cfg, mask2d)
    bce_loss = BceLoss(cfg, mask2d)
    