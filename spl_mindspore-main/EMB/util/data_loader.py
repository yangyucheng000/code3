import mindspore
import mindspore.dataset as ds
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from util.data_util import pad_seq, pad_char_seq, pad_video_seq, load_video_features
from util.args_parser import gconfigs

class Dataset:
    def __init__(self, dataset, video_features):
        self.length = len(dataset)
        self.dataset = dataset

        print('Load video features')
        video_features = [video_features[record['vid']] for record in dataset]
        vfeats, vfeat_lens = pad_video_seq(video_features, max_length=gconfigs.max_pos_len)
        self.vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
        self.vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )

        print('Load labels')
        s_inds, e_inds = [record['s_ind'] for record in dataset], [record['e_ind'] for record in dataset]
        self.s_labels = np.asarray(s_inds, dtype=np.int64)
        self.e_labels = np.asarray(e_inds, dtype=np.int64)

        h_labels = np.zeros(shape=[self.length, gconfigs.max_pos_len], dtype=np.int32)
        extend = 0.1
        for idx in range(self.length):
            st, et = s_inds[idx], e_inds[idx]
            cur_max_len = self.vfeat_lens[idx]
            extend_len = round(extend * float(et - st + 1))
            if extend_len > 0:
                st_ = max(0, st - extend_len)
                et_ = min(et + extend_len, cur_max_len - 1)
                h_labels[idx][st_:(et_ + 1)] = 1
            else:
                h_labels[idx][st:(et + 1)] = 1
        self.h_labels = h_labels
            
        print('Load queries')
        word_ids, char_ids = [record['w_ids'] for record in dataset], [record['c_ids'] for record in dataset]
        word_ids, _ = pad_seq(word_ids, max_length=gconfigs.max_pos_len)
        self.word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        char_ids, _ = pad_char_seq(char_ids, max_length=gconfigs.max_pos_len)
        self.char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)


    def __getitem__(self, idx):
        return idx, self.vfeats[idx], self.vfeat_lens[idx], self.word_ids[idx], self.char_ids[idx], self.s_labels[idx], self.e_labels[idx], self.h_labels[idx]

    def __len__(self):
        return self.length
    
    def get_record(self, idxs):
        return [self.dataset[idx] for idx in idxs]
    


def get_train_loader(dataset, video_features, configs):
    train_set = Dataset(dataset=dataset, video_features=video_features)
    train_loader = ds.GeneratorDataset(train_set, column_names=['indexes', 'vfeats', 'vfeat_lens', 'word_ids', 'char_ids', 's_labels', 'e_labels', 'h_labels'], shuffle=True)
    train_loader = train_loader.batch(batch_size=configs.batch_size, drop_remainder=False)
    return train_loader


def get_test_loader(dataset, video_features, configs):
    test_set = Dataset(dataset=dataset, video_features=video_features)
    test_loader = ds.GeneratorDataset(test_set, column_names=['indexes', 'vfeats', 'vfeat_lens', 'word_ids', 'char_ids', 's_labels', 'e_labels', 'h_labels'], shuffle=False)
    test_loader = test_loader.batch(batch_size=configs.batch_size)
    return test_loader
