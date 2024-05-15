import mindspore
import mindspore.ops.function as F
import mindspore.ops as P
import pickle
import numpy as np
from tqdm import tqdm
import os
import  json
import argparse
import nltk

DATA_PATH={
    'charades': 'data/charades/train.json',
    'activitynet': 'data/activitynet/train.json'
}

def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].astype(np.float32), gt[1].astype(np.float32)
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def nms(moments, scores, thresh):
    ranks = scores.argsort()[::-1]
    scores = scores[ranks]
    moments = moments[ranks]
    suppressed = np.zeros_like(ranks, dtype=np.bool_)
    numel = suppressed.size
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
    return moments[~suppressed], scores[~suppressed]


def nms2(moments, scores, sims, thresh, topk=5):
    reversed_idx = np.arange(0, scores.shape[0], dtype=np.int64)
    final_scores = scores * sims
    ranks = final_scores.argsort()[::-1]
    final_scores = final_scores[ranks]
    # ranks = np.random.permutation(scores.shape[0])
    moments = moments[ranks]
    scores = scores[ranks]
    sims = sims[ranks]
    reversed_idx = reversed_idx[ranks]
    # return moments, scores, sims, reversed_idx
    suppressed = np.zeros_like(ranks, dtype=np.int64)
    numel = suppressed.size
    for i in range(numel - 1):
        if suppressed[i] >= topk:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] += 1
    suppressed = suppressed >= topk
    return moments[~suppressed], scores[~suppressed], sims[~suppressed], reversed_idx[~suppressed]


def enumerate_with_sliding_window(scores, captions, stride, nms_topk, nms_thresh):
    flattened_captions = [s for c in captions for s in c]
    flattened_proposals = []
    flattened_scores = []
    flattened_similarity = []
    max_stride = scores.shape[-1]
    stride = min(stride, max_stride)
    for kernel_size in range(stride, max_stride+1, stride):
        res = F.conv1d(scores.view(-1, 1, scores.shape[-1]), F.ones((1, 1, kernel_size)) / kernel_size).view(scores.shape[0], scores.shape[1], -1)
        res = res - 1.0*(scores.sum(axis=-1, keepdims=True) - res * kernel_size) / (scores.shape[-1] - kernel_size)
        res = res.view(-1, res.shape[-1]).asnumpy()
        proposals = np.arange(0, res.shape[-1], dtype=np.float32)
        proposals = np.stack([proposals, proposals + kernel_size], axis=-1) / scores.shape[-1]
        for idx in range(len(flattened_captions)):
            mask = res[idx] > 0
            if idx >= len(flattened_proposals):
                flattened_proposals.append(proposals[mask])
                flattened_scores.append(res[idx][mask])
                flattened_similarity.append(scores.reshape(len(flattened_captions), -1)[idx].max())
            else:
                flattened_proposals[idx] = np.concatenate([flattened_proposals[idx], proposals[mask]], axis=0)
                flattened_scores[idx] = np.concatenate([flattened_scores[idx], res[idx][mask]], axis=0)

    filtered_captions = []
    filtered_proposals = []
    filtered_scores = []
    filtered_similarity = []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            nms_proposals, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], nms_thresh)
            for k in range(min(nms_topk, nms_scores.shape[0])):
                filtered_captions.append(flattened_captions[idx])
                filtered_similarity.append(flattened_similarity[idx])
                filtered_proposals.append(nms_proposals[k])
                filtered_scores.append(nms_scores[k])

    return filtered_captions, np.stack(filtered_proposals), np.stack(filtered_scores), np.stack(filtered_similarity)


def enumerate_with_events(scores, captions):
    with open('data/activitynet/events.pkl', 'rb') as f:
        events = pickle.load(f)

    scores = scores.asnumpy().reshape(-1, scores.shape[-1])
    flattened_captions = [s for c in captions for s in c]
    flattened_scores = []
    flattened_sims = []
    max_stride = scores.shape[-1]

    for event in events[vid]:
        s, e = event[0] / 128, (event[1] + 1) / 128
        mask = np.arange(max_stride, dtype=np.float32) / max_stride
        mask = (mask >= s) & (mask < e)
        mask = mask.astype(np.float32)
        if (1 - mask).sum() == 0:
            res = np.zeros((scores.shape[0],))
        else:
            res = (scores * mask).sum(axis=-1) / mask.sum() - (scores * (1 - mask)).sum(axis=-1) / (1 - mask).sum()
        flattened_scores.append(res)
        flattened_sims.append(np.ones_like((scores).max(axis=-1)))
    
    flattened_scores = np.stack(flattened_scores, axis=-1)
    flattened_sims = np.stack(flattened_sims, axis=-1)
    max_idx = flattened_scores.argmax(axis=-1)

    filtered_captions = []
    filtered_proposals = []
    filtered_scores = []
    filtered_sims = []
    for idx in range(len(flattened_captions)):
        filtered_captions.append(flattened_captions[idx])
        filtered_proposals.append(events[vid][int(max_idx[idx])])
        filtered_scores.append(flattened_scores[idx, max_idx[idx]])
        filtered_sims.append(flattened_sims[idx, max_idx[idx]])

    return filtered_captions, np.array(filtered_proposals), np.array(filtered_scores), np.array(filtered_sims)


def generate_proposal(vid, stride, nms_thresh=0.3, nms_topk=1, args=None):
    caption_path = os.path.join(args.caption_path, vid+'.pkl')
    caption_feature_path = os.path.join(args.caption_feat_path, vid+'.npy')
    video_feature_path = os.path.join(args.video_feat_path, vid+'.npy')

    try:
        with open(caption_path, 'rb') as f:
            captions = pickle.load(f)
        with open(caption_feature_path, 'rb') as f:
            caption_features = np.load(f)
        with open(video_feature_path, 'rb') as f:
            video_features = np.load(f)
    except:
        return [], [], [], []
    
    verb_filt = F.zeros((len(captions), len(captions[0])))
    for i in range(len(captions)):
        for j in range(len(captions[i])):
            flag = False
            for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(captions[i][j])):
                if 'VB' in tag:
                    if word not in ['is', 'am', 'are', 'was', 'were', 'being', 'been', 'to be', 'be']:
                        flag = True
                        break
            if flag:
                verb_filt[i][j] = 1
    
    v1 = P.L2Normalize(axis=-1)(mindspore.tensor(caption_features).view(-1, caption_features.shape[-1]))
    v2 = P.L2Normalize(axis=-1)(mindspore.tensor(video_features))
    scores = (v1 @ v2.T).reshape(caption_features.shape[0], caption_features.shape[1], -1)
    scores = scores / 2 + 0.5
    scores = scores * verb_filt.unsqueeze(-1)

    if args.dataset == 'charades':
        return enumerate_with_sliding_window(scores, captions, stride, nms_topk, nms_thresh)
    elif args.dataset == 'activitynet':
        return enumerate_with_events(scores, captions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='charades', type=str, choices=['charades', 'activitynet'])
    parser.add_argument('--video_feat_path', required=True, type=str)
    parser.add_argument('--caption_feat_path', required=True, type=str)
    parser.add_argument('--caption_path', required=True, type=str)
    parser.add_argument('--num_stnc', default=10, type=int)
    parser.add_argument('--stride', default=20, type=int)
    parser.add_argument('--prop_topk', default=3, type=int)
    parser.add_argument('--stnc_th', default=0.7, type=float)
    parser.add_argument('--stnc_topk', default=3, type=int)
    args = parser.parse_args()
    
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target='GPU')

    with open(DATA_PATH[args.dataset]) as f:
        data = json.load(f)

    new_data = []
    topk = args.num_stnc
    for vid in tqdm(data.keys()):
        captions, proposals, scores, sims = generate_proposal(vid, stride=args.stride, nms_topk=args.prop_topk, args=args)
        if (len(captions) == 0):
            continue
        nms_proposals, nms_scores, nms_sims, reversed_idx = nms2(proposals, scores, sims, thresh=args.stnc_th, topk=args.stnc_topk)
        for idx in range(min(topk, nms_proposals.shape[0])):
            new_data.append([vid, data[vid]['duration'], (nms_proposals[idx] * data[vid]['duration']).tolist(), captions[reversed_idx[idx]]])

    if args.dataset == 'charades':
        with open('EMB/data/dataset/charades/charades_sta_train_pseudo.txt', 'w') as f:
            for vid, duration, (s, t), query in new_data:
                print('%s %.2f %.2f##%s'%(vid, s, t, query.strip()), file=f)
    elif args.dataset == 'activitynet':
        tmp = {}
        for vid, duration, (s, t), query in new_data:
            if vid not in tmp:
                tmp[vid] = {'duration': duration, 'timestamps': [], 'sentences': []} 
            tmp[vid]['timestamps'].append([s / 127, t / 127])
            tmp[vid]['sentences'].append(query)
        with open('EMB/data/dataset/activitynet/train_pseudo.json', 'w') as f:
            json.dump(tmp, f)
