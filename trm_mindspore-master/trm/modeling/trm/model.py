import mindspore
from mindspore import nn
from mindspore import ops
from .featpool import build_featpool   # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import build_contrastive_loss,build_bce_loss
from .text_encoder import build_text_encoder,pad_sequence
from .proposal_conv import build_proposal_conv
import random
from loguru import logger
import numpy as np

def packed_str_batch(sequence):
    # logger.info(f"sequence: { sequence.tolist()}")
    ret =[]
    for seq in sequence.tolist():
        if isinstance(seq,list):
            row = []
            for i,s in enumerate(seq):
                t = s.replace('#','')
                if len(t)>0:
                    row.append(t)
            if len(row)>0:
                ret.append(row)
                # logger.info(f's: {s}')
        else:
            t = seq.replace('#','')
            if len(t)>0:
                ret.append(t)
            # logger.info(f'seq: {type(seq)}')
    # logger.info(f'ret: {ret}')
    return ret
def pack_batches(batches):
    """
    Arguments:
        batches: dict
    Returns:
        packed_batches: dict
    """
    packed_batches = {}
    packed_batches['feature'] = batches['feature'].float()
    packed_batches['sen_len'] = batches['sen_len'].long()
    packed_batches['index'] = batches['index'].long()
    packed_batches['duration'] = batches['duration'].float()
    query,word_len,iou2d,moment,phrase,sentence = [],[],[],[],[],[]
    # logger.info(f"phrase: {batches['phrase'].asnumpy()[0][0]}")
    for q , sen_l,w_l,i2d,m,p,s in zip(batches['query'], batches['sen_len'],batches['word_len'],batches['iou2d'],batches['moment'],batches['phrase'].asnumpy(),batches['sentence'].asnumpy()):
        word_len.append(w_l[:sen_l].long())
        query.append(q[:sen_l].long())
        iou2d.append(i2d[:sen_l].float())
        moment.append(m[:sen_l].float())
        # logger.info(f"m[:sen_l].float(): {m[:sen_l].float().shape}")
        phrase.append( packed_str_batch(p))
        sentence.append(packed_str_batch(s))
    packed_batches['query'] = query
    packed_batches['word_len'] = word_len
    packed_batches['iou2d'] = iou2d
    packed_batches['moment'] = moment
    packed_batches['phrase'] = phrase
    packed_batches['sentence'] = sentence
    # logger.info(f"packed query: {[q.shape for q in query]}")
    # logger.info(f"packed word_len: {word_len}")
    # logger.info(f"packed moment: {[i.shape for i in moment]}")
    return packed_batches

class TRM(nn.Cell):
    def __init__(self, cfg):
        super(TRM, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.TRM.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.TRM.TEXT_ENCODER.NAME
        self.use_score_map_loss = cfg.MODEL.TRM.LOSS.USE_SCORE_MAP_LOSS
        self.cfg = cfg.MODEL.TRM
        self.thresh = cfg.MODEL.TRM.LOSS.THRESH
        self.norm1 = ops.L2Normalize(axis=1)
        self.norm0 = ops.L2Normalize(axis=0)
        self.w = self.cfg.RESIDUAL

    # def  construct(self,feature,query,word_len,iou2d,moment,phrase,sentence,sen_len,cur_epoch=1):
    def construct(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
        # print(batches)
        # backbone
        # packed data
        batches = pack_batches(batches)
        ious2d = batches['iou2d']
        # logger.info(f"ious2d: {ious2d.shape},feature: {batches['feature'].shape}, query: {batches['query'].shape}, word_len: {batches['word_len']}, num_sentence: {batches['sen_len']}")
        assert len(ious2d) == batches['feature'].shape[0]
        
        for idx, (iou, sent) in enumerate(zip(ious2d, batches['query'])):
            assert iou.shape[0] == sent.shape[0]
            assert iou.shape[0] == batches['sen_len'][idx]
        # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128
        feats = self.featpool(batches['feature'])
        # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d = self.feat2d(feats)
        # two visual features using in different branches, both [B*C*T*T]
        map2d, map2d_iou = self.proposal_conv(map2d)
        # logger.info(f'map2d : {map2d.shape}, map2d_iou: {map2d_iou.shape}')
        # two features using in different branches, both list(B)-[num_sent*C]
        # logger.info(batches)
        # sent_feat, sent_feat_iou, phrase_feat, phrase_feat_iou, phrase_weight, phrase_masks
        
        if self.text_encoder.use_phrase:
            # logger.info(f'self.text_encoder.use_phrase: {self.text_encoder.use_phrase}')
            sent_feat, sent_feat_iou, phrase_feat, phrase_feat_iou, phrase_weight, phrase_mask = self.text_encoder.encode_sentences(batches['sentence'], batches['phrase'])
        else:
            # logger.info(f'self.text_encoder.use_phrase: {self.text_encoder.use_phrase}')
            sent_feat, sent_feat_iou = self.text_encoder(batches['query'], batches['word_len'])
            phrase_feat, phrase_feat_iou, phrase_weight, phrase_mask = mindspore.Tensor(0.0), mindspore.Tensor(0.0), mindspore.Tensor(0.0), mindspore.Tensor(0.0)
        # inference
        contrastive_scores = []
        iou_scores = []
        phrase_iou_scores = []
        phrase_score_map = []
        phrase_score_map_mask = []
        # logger.info(f'map2d : {map2d.shape}')
        _, T, _ = map2d[0].shape
        # logger.info(f'sent_feat_iou: {sent_feat_iou[0].shape}')
        # logger.info(f'init moment : {[m.shape for m in batches["moment"] ]}')
        for i, sf_iou in enumerate(sent_feat_iou):  # sent_feat_iou: [num_sent x C] (len=B)
            # iou part
            if self.cfg.RESIDUAL and self.text_encoder.use_phrase:
                vid_feat_iou = map2d_iou[i]  # C x T x T
                vid_feat_iou_norm = self.norm0(vid_feat_iou)

                phrase_persent_iou_scores = []
                pf_feat_iou = phrase_feat_iou[i]
                for pf_iou_pers in pf_feat_iou:
                    pf_iou_pers_norm = self.norm1(pf_iou_pers) # max_p * C
                    iou_score = ops.mm(pf_iou_pers_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.shape[0], -1)).reshape(-1, T, T) # max_p x T x T
                    phrase_persent_iou_scores.append(iou_score)

                phrase_persent_iou_scores = ops.stack(phrase_persent_iou_scores) # num_sent*max_p*T*T
                phrase_score_map.append((phrase_persent_iou_scores*10).sigmoid() * self.feat2d.mask2d)
                phrase_score_map_mask.append(phrase_mask[i])

                T_num = phrase_persent_iou_scores.shape[2]
                phrase_persent_iou_scores = phrase_persent_iou_scores.reshape(phrase_persent_iou_scores.shape[0], phrase_persent_iou_scores.shape[1], -1)
                phrase_w = phrase_weight[i].unsqueeze(1) # num_sent*1*max_p
                # phrase_w.clamp(0, 0.0001) / phrase_w.clamp(0, 0.0001).sum(dim=-1, keepdim=True)
                phrase_att_iou_scores = phrase_w @ phrase_persent_iou_scores
                phrase_att_iou_scores = phrase_att_iou_scores.reshape(phrase_att_iou_scores.shape[0], T_num, -1) #num*T*T

                sf_iou_norm = self.norm1(sf_iou) # num_sent x C
                sf_iou_score = ops.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.shape[0], -1)).reshape(-1, T, T)  # num_sent x T x T

                sf_iou_score = ((sf_iou_score + self.w * phrase_att_iou_scores) * 10).sigmoid() * self.feat2d.mask2d 
                # sf_iou_score = (((1 - self.w) * sf_iou_score + self.w * phrase_att_iou_scores) * 10).sigmoid() * self.feat2d.mask2d
                # print(phrase_weight[i].shape)
                # print(sf_iou_score.shape)
                # print(phrase_att_iou_scores.shape)
                # sf_iou_score = phrase_weight[i][:, 0].view(-1, 1, 1) * sf_iou_score + phrase_att_iou_scores
                # sf_iou_score = (sf_iou_score * 10).sigmoid() * self.feat2d.mask2d
                iou_scores.append(sf_iou_score)
                
                # pf_feat_iou = (pf_feat_iou * phrase_weight[i]).sum(dim=1)
                # sf_iou_norm = F.normalize(sf_iou + pf_feat_iou, dim=1) # num_sent x C
                # sf_iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T

                # iou_scores.append((sf_iou_score*10).sigmoid() * self.feat2d.mask2d)
            else:
                vid_feat_iou = map2d_iou[i]  # C x T x T
                vid_feat_iou_norm = self.norm0(vid_feat_iou)
                sf_iou_norm = self.norm1(sf_iou) # num_sent x C
                iou_score = ops.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.shape[0], -1)).reshape(-1, T, T)  # num_sent x T x T
                iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)
            
                if self.text_encoder.use_phrase:
                    phrase_persent_iou_scores = []
                    pf_feat_iou = phrase_feat_iou[i]
                    for pf_iou_pers in pf_feat_iou:
                        pf_iou_pers_norm = self.norm1(pf_iou_pers) # max_p * C
                        iou_score = ops.mm(pf_iou_pers_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.shape[0], -1)).reshape(-1, T, T) # max_p x T x T
                        phrase_persent_iou_scores.append(iou_score * self.feat2d.mask2d)

                    phrase_persent_iou_scores = ops.stack(phrase_persent_iou_scores) # num_sent*max_p*T*T
                    phrase_score_map.append((phrase_persent_iou_scores*10).sigmoid() * self.feat2d.mask2d)
                    phrase_score_map_mask.append(phrase_mask[i])
                    T_num = phrase_persent_iou_scores.shape[2]
                    phrase_persent_iou_scores = phrase_persent_iou_scores.reshape(phrase_persent_iou_scores.shape[0], phrase_persent_iou_scores.shape[1], -1)
                    
                    phrase_w = phrase_weight[i].unsqueeze(1) # num_sent*1*max_p
                    phrase_att_iou_scores = phrase_w @ phrase_persent_iou_scores
                    phrase_att_iou_scores = phrase_att_iou_scores.reshape(phrase_att_iou_scores.shape[0], T_num, -1) #num*T*T

                    phrase_iou_scores.append((phrase_att_iou_scores * 10).sigmoid() * self.feat2d.mask2d)
                    # pf_feat_iou = (pf_feat_iou * phrase_weight[i]).sum(dim=1)
                    # pf_feat_iou_norm = F.normalize(pf_feat_iou, dim=1) # num_sent x C
                    # iou_score = torch.mm(pf_feat_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T) # max_p x T x T
                    # phrase_iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)

        # phrase contrastive

        if self.text_encoder.use_phrase and self.cfg.LOSS.CONTRASTIVE:
            # logger.info(f'self.text_encoder.use_phrase: {self.text_encoder.use_phrase}, self.cfg.LOSS.CONTRASTIVE: {self.cfg.LOSS.CONTRASTIVE}')
            contrast_neg_vid = []
            contrast_neg_phr = []
            contrast_neg_mask = []
            BS = len(sent_feat_iou)
            for i in range(BS):  # sent_feat_iou: [num_sent x C] (len=B)
                neg_idx = random.randint(0, BS-1)
                while neg_idx == i:
                    neg_idx = random.randint(0, BS-1)

                # negative video
                vid_feat = map2d_iou[neg_idx]  # C x T x T
                vid_feat_norm = self.norm0(vid_feat)
                phrase_persent_scores = []
                pf_feat = phrase_feat_iou[i]
                for pf_pers in pf_feat:
                    pf_pers_norm = self.norm1(pf_pers) # max_p * C
                    score = ops.mm(pf_pers_norm, vid_feat_norm.reshape(vid_feat_norm.shape[0], -1)).reshape(-1, T, T) # max_p x T x T
                    phrase_persent_scores.append(score)
                phrase_persent_scores = ops.stack(phrase_persent_scores) # num_sent*max_p*T*T
                contrast_neg_vid.append((phrase_persent_scores*10).sigmoid() * self.feat2d.mask2d)

                vid_feat = map2d_iou[i]  # C x T x T
                vid_feat_norm = self.norm0(vid_feat)
                phrase_persent_scores = []
                pf_feat = phrase_feat_iou[neg_idx]
                for pf_pers in pf_feat:
                    pf_pers_norm = self.norm1(pf_pers) # max_p * C
                    score = ops.mm(pf_pers_norm, vid_feat_norm.reshape(vid_feat_norm.shape[0], -1)).reshape(-1, T, T) # max_p x T x T
                    phrase_persent_scores.append(score)
                phrase_persent_scores = ops.stack(phrase_persent_scores) # num_sent*max_p*T*T
                contrast_neg_phr.append((phrase_persent_scores*10).sigmoid() * self.feat2d.mask2d)
                contrast_neg_mask.append(phrase_mask[neg_idx])
        else:
            # logger.info(f'self.text_encoder.use_phrase: {self.text_encoder.use_phrase}, self.cfg.LOSS.CONTRASTIVE: {self.cfg.LOSS.CONTRASTIVE}')
            contrast_neg_vid = mindspore.Tensor(0.0)
            contrast_neg_phr = None
            contrast_neg_mask = None

        if self.text_encoder.use_phrase and self.use_score_map_loss:
            # logger.info(f'self.text_encoder.use_phrase: {self.text_encoder.use_phrase}, self.use_score_map_loss: {self.use_score_map_loss}')
            phrase_score_map = pad_sequence(phrase_score_map, batch_first=True) # B, NS, NP, T, T
            B, NS, NP, T1, T2 = phrase_score_map.shape
            phrase_score_map_mask = pad_sequence(phrase_score_map_mask, batch_first=True) # B, NS, NP
            # logger.info(f'phrase_score_map_mask: {phrase_score_map_mask.shape}, phrase_score_map: {phrase_score_map.shape}')
            phrase_score_map_mask_exp = phrase_score_map_mask.reshape(B, NS, NP, 1, 1).broadcast_to((B, NS, NP, T1, T2))
            phrase_iou2d = pad_sequence(ious2d, batch_first=True) # B, NS, T, T
            phrase_iou2d_exp = phrase_iou2d.unsqueeze(2).broadcast_to((B, NS, NP, T1, T2))

            scoremap_loss_pos, _ = phrase_score_map.masked_fill(phrase_iou2d_exp * self.feat2d.mask2d < self.thresh, -1e9).view(B, NS, NP, -1).max(axis=-1,return_indices=True)
            # scoremap_loss_pos, idx = scoremap_loss_pos.masked_fill(phrase_score_map_mask==0, 1e9).min(dim=-1)
            if self.cfg.LOSS.USE_FOCAL_LOSS:
                scoremap_loss_pos = -scoremap_loss_pos.log() * (1 - scoremap_loss_pos).pow(2)
            else:
                scoremap_loss_pos = -scoremap_loss_pos.log()
            scoremap_loss_pos = ops.where(ops.isnan(scoremap_loss_pos), ops.full_like(scoremap_loss_pos, 0), scoremap_loss_pos)
            scoremap_loss_pos = ops.where(ops.isinf(scoremap_loss_pos), ops.full_like(scoremap_loss_pos, 0), scoremap_loss_pos)
            scoremap_loss_pos = ops.sum(scoremap_loss_pos * phrase_score_map_mask) / ops.sum(phrase_score_map_mask)
            # mask = phrase_score_map_mask.gather(index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
            # scoremap_loss_pos = torch.sum(scoremap_loss_pos * mask) / torch.sum(mask)

            if self.cfg.LOSS.CONTRASTIVE:
                # logger.info(f'self.cfg.LOSS.CONTRASTIVE: {self.cfg.LOSS.CONTRASTIVE}')
                contrast_neg_vid = pad_sequence(contrast_neg_vid, batch_first=True) # B, NS, NP, T, T
                B, NS, NP, T1, T2 = contrast_neg_vid.shape
                scoremap_loss_neg_vid, _ = contrast_neg_vid.masked_select(self.feat2d.mask2d==1).view(B, NS, NP, -1).max(axis=-1,return_indices=True)
                # scoremap_loss_neg_vid, idx = scoremap_loss_neg_vid.masked_fill(phrase_score_map_mask==0, 1e9).min(dim=-1)
                if self.cfg.LOSS.USE_FOCAL_LOSS:
                    scoremap_loss_neg_vid = -(1 - scoremap_loss_neg_vid).log() * scoremap_loss_neg_vid.pow(2)
                else:
                    scoremap_loss_neg_vid = -(1 - scoremap_loss_neg_vid).log()
                scoremap_loss_neg_vid = ops.where(ops.isnan(scoremap_loss_neg_vid), ops.full_like(scoremap_loss_neg_vid, 0), scoremap_loss_neg_vid)
                scoremap_loss_neg_vid = ops.where(ops.isinf(scoremap_loss_neg_vid), ops.full_like(scoremap_loss_neg_vid, 0), scoremap_loss_neg_vid)
                scoremap_loss_neg_vid = ops.sum(scoremap_loss_neg_vid * phrase_score_map_mask) / ops.sum(phrase_score_map_mask)
                # mask = phrase_score_map_mask.gather(index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
                # scoremap_loss_neg_vid = torch.sum(scoremap_loss_neg_vid * mask) / torch.sum(mask)
                # logger.info('start loss')
                contrast_neg_phr = pad_sequence(contrast_neg_phr, batch_first=True) # B, NS, NP, T, T
                B, NS, NP, T1, T2 = contrast_neg_phr.shape
                contrast_neg_mask = pad_sequence(contrast_neg_mask, batch_first=True) # B, NS, NP
                scoremap_loss_neg_phr, _ = contrast_neg_phr.masked_select(self.feat2d.mask2d==1).view(B, NS, NP, -1).max(axis=-1,return_indices=True)
                # scoremap_loss_neg_phr, idx = scoremap_loss_neg_phr.masked_fill(contrast_neg_mask==0, 1e9).min(dim=-1)
                if self.cfg.LOSS.USE_FOCAL_LOSS:
                    scoremap_loss_neg_phr = -(1 - scoremap_loss_neg_phr).log() * scoremap_loss_neg_phr.pow(2)
                else:
                    scoremap_loss_neg_phr = -(1 - scoremap_loss_neg_phr).log()
                scoremap_loss_neg_phr = ops.where(ops.isnan(scoremap_loss_neg_phr), ops.full_like(scoremap_loss_neg_phr, 0), scoremap_loss_neg_phr)
                scoremap_loss_neg_phr = ops.where(ops.isinf(scoremap_loss_neg_phr), ops.full_like(scoremap_loss_neg_phr, 0), scoremap_loss_neg_phr)
                scoremap_loss_neg_phr = ops.sum(scoremap_loss_neg_phr * contrast_neg_mask) / ops.sum(contrast_neg_mask)
                # mask = contrast_neg_mask.gather(index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
                # scoremap_loss_neg_phr = torch.sum(scoremap_loss_neg_phr * mask) / torch.sum(mask)

                scoremap_loss_neg = (scoremap_loss_neg_vid + scoremap_loss_neg_phr) / 2
            # else:
                scoremap_loss_exc, _ = phrase_score_map.masked_fill(phrase_score_map_mask_exp == 0, 1e9).min(axis=2,return_indices=True)
                if self.cfg.LOSS.USE_FOCAL_LOSS:
                    scoremap_loss_exc = - (1 - scoremap_loss_exc).log() * scoremap_loss_exc.pow(2)
                else:
                    scoremap_loss_exc = - (1 - scoremap_loss_exc).log()
                scoremap_loss_exc = ops.where(ops.isnan(scoremap_loss_exc), ops.full_like(scoremap_loss_exc, 0), scoremap_loss_exc)
                scoremap_loss_exc = ops.where(ops.isinf(scoremap_loss_exc), ops.full_like(scoremap_loss_exc, 0), scoremap_loss_exc)
                scoremap_loss_exc = ops.sum(scoremap_loss_exc * (phrase_iou2d < self.thresh).long() * self.feat2d.mask2d) / ops.sum((phrase_iou2d < self.thresh).long() * self.feat2d.mask2d)
            else:
                # logger.info(f'self.cfg.LOSS.CONTRASTIVE: {self.cfg.LOSS.CONTRASTIVE}')
                scoremap_loss_pos = mindspore.Tensor(0.0)
                scoremap_loss_neg = mindspore.Tensor(0.0)
                scoremap_loss_exc = mindspore.Tensor(0.0)
        else:
            # logger.info(f'self.text_encoder.use_phrase: {self.text_encoder.use_phrase}, self.use_score_map_loss: {self.use_score_map_loss}')
            contrast_neg_vid = mindspore.Tensor(0.0)
            scoremap_loss_pos = mindspore.Tensor(0.0)
            scoremap_loss_neg = mindspore.Tensor(0.0)
            scoremap_loss_exc = mindspore.Tensor(0.0)
        # scoremap_loss_pos = torch.tensor(0.0).cuda()
        # scoremap_loss_neg = torch.tensor(0.0).cuda()
        # scoremap_loss_exc = torch.tensor(0.0).cuda()

        # loss
        # if self.training:
        # import pdb
        # pdb.set_trace()
        # logger.info('start loss iou')
        if self.text_encoder.use_phrase and not self.cfg.RESIDUAL:
            loss_iou_phrase = self.iou_score_loss(ops.cat(phrase_iou_scores, axis=0), ops.cat(ious2d, axis=0), cur_epoch)
            if not self.cfg.LOSS.PHRASE_ONLY:
                loss_iou_stnc = self.iou_score_loss(ops.cat(iou_scores, axis=0), ops.cat(ious2d, axis=0), cur_epoch)
            else:
                loss_iou_stnc = mindspore.Tensor(0.0)
        else:
            loss_iou_stnc = self.iou_score_loss(ops.cat(iou_scores, axis=0), ops.cat(ious2d, axis=0), cur_epoch)
            loss_iou_phrase = mindspore.Tensor(0.0)
        # logger.info(f'start loss: map2d{map2d.shape} ious2d{[i.shape for i in ious2d]}')
        # logger.info(f'start loss moment : {[m.shape for m in batches["moment"]]}')
        loss_vid, loss_sent = self.contrastive_loss(map2d, sent_feat, ious2d, None, None, batches['moment'])
        # loss = (loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc)
        # del scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc
        # loss = (loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc)
        # logger.info('start contrastive_score')
        for i, sf in enumerate(sent_feat):
            # contrastive part
            vid_feat = map2d[i, ...]  # C x T x T
            vid_feat_norm = self.norm0(vid_feat)

            sf_norm = self.norm1(sf)  # num_sent x C
            _, T, _ = vid_feat.shape
            contrastive_score = ops.mm(sf_norm, vid_feat_norm.reshape(vid_feat.shape[0], -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
            contrastive_scores.append(contrastive_score)
        
        if self.text_encoder.use_phrase and not self.cfg.RESIDUAL:
            
            # return map2d_iou, sent_feat_iou, contrastive_scores, phrase_iou_scores  # first two maps for visualization
            if self.cfg.LOSS.PHRASE_ONLY:
                return contrastive_scores, phrase_iou_scores, loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc  # first two maps for visualization
            else:
                # return map2d_iou, sent_feat_iou, contrastive_scores, [(s + p) / 2 for s, p in zip(iou_scores, phrase_iou_scores)], loss # first two maps for visualization
                return  contrastive_scores, iou_scores, loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc # first two maps for visualization
        else:
            # logger.info(f'end loss, {self.text_encoder.use_phrase}, {self.cfg.RESIDUAL}, {self.cfg.LOSS.PHRASE_ONLY}')
            return  contrastive_scores, iou_scores, loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc



    
    