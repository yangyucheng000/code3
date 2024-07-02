'''
The visual and textual data provider and corresponding collate_fn are same for both per-vieo and per-query setting
The train dataset and corresponding collate_fn are different for per-video and per-query setting, 
which should be implemented in the specific task's folder
'''


import mindspore as ms
import mindspore.ops as ops
import numpy as np
import mindspore.numpy as msnp
import re
import h5py
from utils.basic_utils import load_jsonl


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = ms.Tensor.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = ops.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = ops.minimum(ops.round(idxs).astype(ms.int32), ms.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(ops.mean(visual_input[s_idx:e_idx], axis=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = ops.stack(new_visual_input, axis=0).asnumpy()

    return new_visual_input


def uniform_feature_sampling_ori(features, max_len): # numpy version
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features
# mindspore version(may be faster)
def uniform_feature_sampling(visual_input, map_size):
    num_clips = visual_input.shape[0]
    if map_size is None or num_clips <= map_size:
        return visual_input
    visual_input = ms.Tensor.from_numpy(visual_input)
    num_sample_clips = map_size
    idxs = ops.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = ops.minimum(ops.round(idxs).astype(ms.int32), ms.Tensor([num_clips - 1]))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(ops.mean(visual_input[s_idx:e_idx], axis=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = ops.stack(new_visual_input, axis=0).asnumpy()

    return new_visual_input



def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)



def collate_train(clip_video_features, frame_video_features, cap_feat, idxs, cap_id, video_id, batch_info):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # zipped = zip(clip_video_features, frame_video_features, cap_feat, idxs, cap_id, video_id)
    # data = list(zipped)
    # # Sort a data list by caption length
    # if data[0][1] is not None:
    #     data.sort(key=lambda x: len(x[1]), reverse=True)
    # # clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, words_ids_, words_feats_, weightss_, words_lens_ = zip(*data) # with rec
    # clip_video_features, frame_video_features, cap_feat, idxs, cap_id, video_id = zip(*data)
    # clip_video_features = list(clip_video_features)
    # frame_video_features = list(frame_video_features)
    # cap_feat = list(cap_feat)
    
    #videos
    clip_videos = np.stack(clip_video_features, axis=0)
    clip_videos = ms.Tensor(clip_videos).astype(ms.float32)

    for idx, vec in enumerate(frame_video_features):
        frame_video_features[idx] = ms.Tensor(vec)
    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = ops.zeros((len(frame_video_features), max(video_lengths), frame_vec_len), dtype=ms.float32)
    videos_mask = ops.zeros((len(frame_video_features), max(video_lengths)), dtype=ms.float32)
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    for idx, vec in enumerate(cap_feat):
        cap_feat[idx] = ms.Tensor(vec)
    feat_dim = cap_feat[0].shape[-1]

    roberta_length = [len(e) for e in cap_feat]

    padded_roberta_feat = ops.zeros((len(cap_feat), max(roberta_length), feat_dim), dtype=ms.float32)
    roberta_mask = ops.zeros((len(cap_feat), max(roberta_length)), dtype=ms.float32)

    for index, cap in enumerate(cap_feat):
        end = roberta_length[index]
        padded_roberta_feat[index, :end, :] = cap[:end, :]
        roberta_mask[index, :end] = 1.0

    # return clip_videos, frame_videos, videos_mask, padded_roberta_feat, roberta_mask

    return dict(clip_video_features=clip_videos,
                frame_video_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=padded_roberta_feat,
                text_mask=roberta_mask,
                )


def collate_frame_val(clip_video_features, frame_video_features, idxs, video_ids, batch_info):
    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # videos
    clip_videos = np.stack(clip_video_features, axis=0)
    clip_videos = ms.Tensor(clip_videos).astype(ms.float32)

    for idx, vec in enumerate(frame_video_features):
        frame_video_features[idx] = ms.Tensor(vec)
    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = ops.zeros((len(frame_video_features), max(video_lengths), frame_vec_len), dtype=ms.float32)
    videos_mask = ops.zeros((len(frame_video_features), max(video_lengths)), dtype=ms.float32)
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    # todo: process idxs and video_ids

    return dict(clip_videos=clip_videos, frame_videos=frame_videos, videos_mask=videos_mask, idxs=idxs, video_ids=video_ids)


def collate_text_val(captions,idxs, cap_ids, batch_info):
    # zipped = zip(captions,idxs, cap_ids)
    # data = list(zipped)
    # if data[0][0] is not None:
    #     data.sort(key=lambda x: len(x[0]), reverse=True)
    # captions,idxs, cap_ids = zip(*data)
    # captions = list(captions)
    # idxs = list(idxs)
    # cap_ids = list(cap_ids)
    if captions[0] is not None:
        for idx, vec in enumerate(captions):
            captions[idx] = ms.Tensor(vec)
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = ops.zeros((len(captions), max(lengths), captions[0].shape[-1]), dtype=ms.float32)
        words_mask = ops.zeros((len(captions), max(lengths)), dtype=ms.float32)
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None

    return dict(target=target, words_mask=words_mask, idxs=idxs, cap_ids=cap_ids)



class Dataset4Training():
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, video_feat_path, text_feat_path, opt, video2frames=None): # , vocab
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.cap2vid = {}
        # self.video2frames = video2frames

        self.cap_data = load_jsonl(cap_file)
        self.data_ratio = opt.train_data_ratio
        if self.data_ratio != 1:
            n_examples = int(len(self.cap_data) * self.data_ratio)
            self.cap_data = self.cap_data[:n_examples]
            print("Using {}% of the data for training: {} examples".format(self.data_ratio * 100, n_examples))
        
        for idx, item in enumerate(self.cap_data):
            # vid_id, duration, st_ed, cap_id, caption = item # v1
            vid_id = item['vid_name']
            # query_id = item['desc_id']
            # st_ed = item['ts']
            cap_id = item['desc_id']
            caption = item['desc']
            self.captions[cap_id] = caption
            self.cap_ids.append(cap_id)
            self.cap2vid[cap_id] = vid_id
            if vid_id not in self.video_ids:
                self.video_ids.append(vid_id)
            if vid_id in self.vid_caps:
                self.vid_caps[vid_id].append(cap_id)
            else:
                self.vid_caps[vid_id] = []
                self.vid_caps[vid_id].append(cap_id)
            

        # self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path
        # new vid feat
        self.visual_feat = h5py.File(video_feat_path, 'r')

        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l

        # self.open_file = False
        # self.length = len(self.vid_caps) # per video
        self.length = len(self.cap_data) # per query

        self.text_feat = h5py.File(self.text_feat_path, 'r')

        # add
        self.frame_feat_sample = opt.frame_feat_sample



    def __getitem__(self, index):

        cap_id = self.cap_ids[index]
        video_id = self.cap2vid[cap_id]


        # new vid feat:
        frame_vecs = np.array(self.visual_feat[video_id][...], dtype=np.float32)

        clip_video_feature = average_to_fixed_length(frame_vecs, self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        # clip_video_feature = ms.Tensor(clip_video_feature).unsqueeze(0) # change collate_fn

        if self.frame_feat_sample == 'fixed':
            frame_video_feature = average_to_fixed_length(frame_vecs, self.max_ctx_len)
        else:
            frame_video_feature = uniform_feature_sampling(frame_vecs, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        # frame_video_feature = ms.Tensor(frame_video_feature)

        # text
        cap_feat = np.array(self.text_feat[str(cap_id)][...]).astype(np.float32) # changed
        normed_cap_feat = l2_normalize_np_array(cap_feat)
        # print(cap_feat.dtype, normed_cap_feat.dtype)
        # cap_tensor = ms.Tensor.from_numpy(normed_cap_feat)[:self.max_desc_len]
        cap_tensor = normed_cap_feat[:self.max_desc_len]


        return clip_video_feature, frame_video_feature, cap_tensor, index, cap_id, video_id # no rec
        

    def __len__(self):
        return self.length

class VisDataSet4Test():

    def __init__(self, video_feat_path, opt, cap_file):
        self.visual_feat = h5py.File(video_feat_path, 'r')
        self.video_ids = []
        self.vid2duration = {}
        self.vid2idx = {}

        self.cap_data = load_jsonl(cap_file)
        for idx, item in enumerate(self.cap_data):
            # vid_id, duration, st_ed, cap_id, caption = item # v1
            vid_id = item['vid_name']
            duration = item['duration']
            query_id = item['desc_id']

            if vid_id not in self.video_ids:
                self.video_ids.append(vid_id)
            self.vid2duration[vid_id] = float(duration)
            if vid_id not in self.vid2idx:
                self.vid2idx[vid_id] = idx
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        # add
        self.frame_feat_sample = opt.frame_feat_sample

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        # new vid feat:
        frame_vecs = np.array(self.visual_feat[video_id][...]).astype(np.float32)
        
        clip_video_feature = average_to_fixed_length(frame_vecs, self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        # clip_video_feature = ops.Tensor(clip_video_feature).unsqueeze(0)

        if self.frame_feat_sample == 'fixed':
            frame_video_feature = average_to_fixed_length(frame_vecs, self.max_ctx_len)
        else:
            frame_video_feature = uniform_feature_sampling(frame_vecs, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        # frame_video_feature = ms.Tensor(frame_video_feature)

        return clip_video_feature, frame_video_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4Test():
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, opt):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.cap2vid = {}

        self.cap_data = load_jsonl(cap_file)

        self.data_ratio = opt.eval_data_ratio
        if self.data_ratio != 1.0:
            n_examples = int(len(self.cap_data) * self.data_ratio)
            self.cap_data = self.cap_data[:n_examples]
            print("In text dataset: ")
            print("Using {}% of the data for evaluation: {} examples".format(self.data_ratio * 100, n_examples))

        for idx, item in enumerate(self.cap_data):
            # vid_id, duration, st_ed, cap_id, caption = item # v1
            vid_id = item['vid_name']
            # duration = item['duration']
            # st_ed = item['ts']
            cap_id = item['desc_id']
            caption = item['desc']

            self.captions[cap_id] = caption
            self.cap_ids.append(cap_id)
            self.cap2vid[cap_id] = vid_id

        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.open_file = False
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True


        cap_feat = np.array(self.text_feat[str(cap_id)][...], dtype=np.float32)

        # cap_tensor = ms.Tensor(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
        normed_cap_feat = l2_normalize_np_array(cap_feat)
        cap_tensor = normed_cap_feat[:self.max_desc_len]

        return cap_tensor, index, cap_id

    def __len__(self):
        return self.length




if __name__ == '__main__':
    pass


