import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from modules.ml_head import ML_Head
import mindspore.numpy as msnp
from modules.model import JSGNet as BaseModel


class JSGNet(BaseModel):
    def __init__(self, config):
        super(JSGNet, self).__init__(config)

        # ml head
        self.ml_head = ML_Head(config)
        self.reset_parameters() # todo: implement

    def construct(self, clip_video_feat, frame_video_feat, frame_video_mask, query_feat, query_mask): # with rec: , words_id, words_feat, words_len, words_weights; per video:, query_labels


        encoded_frame_feat, vid_proposal_feat, encoded_clip_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)
        nq, np, nv = vid_proposal_feat.shape
        clip_scale_scores, frame_scale_scores, clip_scale_scores_, frame_scale_scores_, var4ml \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, vid_proposal_feat, encoded_clip_feat, #, frame_video_mask,# cross=False,
            return_query_feats=True)

        # clip scale loss
        rt = 0.04 # 0.02
        clip_nce_loss = rt * self.video_nce_criterion(clip_scale_scores_)
        clip_trip_loss = self.get_frame_trip_loss(clip_scale_scores)
        clip_loss = clip_nce_loss + clip_trip_loss

        # attention-based frame loss
        atten_nce_loss = atten_trip_loss = 0
        atten_nce_loss = 0.04 * self.video_nce_criterion(frame_scale_scores_)
        atten_trip_loss = self.get_frame_trip_loss(frame_scale_scores)
        atten_loss = atten_nce_loss + atten_trip_loss

        # clip scale intra-video trip loss
        clip_intra_trip_loss = 0
        intra_margin = self.config.intra_margin
        intra_margin2 = intra_margin - 0.1

        alpha1 = self.config.alpha1
        alpha2 = self.config.alpha2
        alpha3 = self.config.alpha3
        beta1 = self.config.beta1
        beta2 = self.config.beta2

        inter_video_trip_loss = 0
        intra_video_trip_loss = 0
        inter_video_nce_loss=0
        frame_intra_trip_loss = 0

        # moment localization
        sim_based_results = self.ml_head.sim_based_results(var4ml['key_clip_indices'], var4ml['q2c_props_scores'], # query_labels, 
                                                encoded_frame_feat, frame_video_mask, var4ml['video_query']) # var4ml['video_query'].detach()
        frame_selected_scores = sim_based_results['gauss_guided_q2vscore']
        unnormed_frame_selected_scores = sim_based_results['gauss_guided_q2vscore_']
        # inter-vid loss
        inter_video_trip_loss = self.get_frame_trip_loss(frame_selected_scores)
        inter_video_nce_loss = rt * self.video_nce_criterion(unnormed_frame_selected_scores)
        inter_video_CL_loss = inter_video_nce_loss + inter_video_trip_loss

        # frame scale intra-video trip loss
        idx = [i for i in range(nq)]
        matched_neg1_score = msnp.diag(sim_based_results['neg1_selected_ggq2v_score']) #[idx, idx]
        matched_neg2_score = msnp.diag(sim_based_results['neg2_selected_ggq2v_score']) 
        matched_hard_neg_score = msnp.diag(sim_based_results['hard_neg_score'])
        matched_frame_score = msnp.diag(frame_selected_scores)
        # side negative:
        _, frame_intra_trip_loss_1 = self.vector_trip_loss(matched_frame_score, matched_neg1_score, intra_margin)
        _, frame_intra_trip_loss_2 = self.vector_trip_loss(matched_frame_score, matched_neg2_score, intra_margin)
        frame_intra_side_loss = frame_intra_trip_loss_1 + frame_intra_trip_loss_2
        # hard negative:
        _, frame_intra_trip_loss_hard = self.vector_trip_loss(matched_frame_score, matched_hard_neg_score, 0.1)

        frame_intra_trip_loss = beta1 * frame_intra_side_loss + beta2 * frame_intra_trip_loss_hard


        # reconstruction based learning
        pos_rec_loss = 0
        rec_neg_vid_trip_loss = 0
        rec_neg_query_trip_loss = 0
        pos_sl_rec_loss = 0
        intra_vid_rec_cl_loss = 0
        kd_kl_loss = 0
        
        loss = alpha1*clip_loss + alpha2*inter_video_CL_loss + alpha3*frame_intra_trip_loss

       
        return loss
        # {"loss_overall": float(loss), 'clip_nce_loss': clip_nce_loss,
        #               'clip_trip_loss': clip_trip_loss,
        #                 'inter_video_trip_loss': inter_video_trip_loss,
        #                'clip_intra_trip_loss': clip_intra_trip_loss, 'inter_video_nce_loss': inter_video_nce_loss,
        #                'frame_intra_trip_loss': frame_intra_trip_loss, 
        #               'pos_rec_loss':pos_rec_loss, 'rec_neg_vid_trip_loss': rec_neg_vid_trip_loss, 
        #               'rec_neg_query_trip_loss': rec_neg_query_trip_loss, 'pos_sl_rec_loss': pos_sl_rec_loss,
        #               'intra_vid_rec_cl_loss': intra_vid_rec_cl_loss, "kd_kl_loss": kd_kl_loss}


    def get_pred_from_raw_query(self, query_feat, query_mask, 
                                video_proposal_feat=None,
                                video_feat=None,
                                video_feat_mask=None,
                                return_query_feats=False,
                                query_labels=None):


        video_query = self.encode_query(query_feat, query_mask)

        # get clip-level retrieval scores

        clip_scale_scores, key_clip_indices, q2c_props_scores = self.get_clip_scale_scores(
            video_query, video_proposal_feat)
        
        # for ml_head:
        var4ml = dict()
        var4ml['key_clip_indices'] = key_clip_indices
        var4ml['video_query'] = video_query
        var4ml['q2c_props_scores'] = q2c_props_scores

        l2norm = ops.L2Normalize(axis=-1)

        if return_query_feats:
            frame_scale_feat = self.key_clip_guided_attention(video_feat, video_proposal_feat, video_feat_mask, # TODO:try detach frame feat, only optimize clip feat
                                                          key_clip_indices) # , query_labels
            frame_scale_scores = ops.matmul(l2norm(video_query), # video_query.detach()
                                              l2norm(frame_scale_feat).t())
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = ops.matmul(video_query, frame_scale_feat.t()) # video_query.detach()

            return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_, var4ml

        else:
            frame_scale_feat = self.key_clip_guided_attention_in_inference(video_feat, video_proposal_feat, video_feat_mask,
                                                                       key_clip_indices)
            frame_scales_cores_ = ops.mul(l2norm(frame_scale_feat),
                                            l2norm(video_query).unsqueeze(0))
            frame_scale_scores = ops.sum(frame_scales_cores_, dim=-1).swapaxes(1, 0)

            return clip_scale_scores, frame_scale_scores, var4ml # add video_query for wsml_frame_score

