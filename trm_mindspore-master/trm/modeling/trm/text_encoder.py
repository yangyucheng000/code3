from mindspore import nn,Tensor
from mindspore import ops
from mindformers import AutoConfig, BertModel, AutoTokenizer

from mindformers import pipeline

class AttentivePooling(nn.Cell):
    def __init__(self, feat_dim, att_hid_dim):
        super().__init__()

        self.feat_dim = feat_dim
        self.att_hid_dim = att_hid_dim

        self.feat2att = nn.Dense(self.feat_dim, self.att_hid_dim, has_bias=False)
        self.to_alpha = nn.Dense(self.att_hid_dim, 1, has_bias=False)
        # self.fc_phrase = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        # self.fc_sent = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        self.softmax = nn.Softmax(axis=1)

    def construct(self, feats, global_feat, f_masks=None):
        """ Compute attention weights
        Args:
            feats: features where attention weights are computed; [num_sen, num_phr, D]
            global_feat: [num_sen, D]
            f_masks: mask for effective features; [num_sen, num_phr]
        """
        # check inputs
        assert len(feats.shape) == 3
        assert len(global_feat.shape) == 2
        assert f_masks is None or len(f_masks.shape) == 2

        # embedding feature vectors
        # feats = self.fc_phrase(feats)   # [num_sen,num_phr,hdim]
        # global_feat = self.fc_sent(global_feat).unsqueeze(-1) # [num_sen, hdim, 1]
        # alpha = torch.bmm(feats, global_feat) / math.sqrt(self.att_hid_dim) # [num_sen, num_phr, 1]
        # feats = torch.cat([global_feat.unsqueeze(1), feats], dim=1)
        attn_f = self.feat2att(feats)

        # compute attention weights
        dot = ops.tanh(attn_f)        # [num_sen,num_phr,hdim]
        alpha = self.to_alpha(dot)      # [num_sen,num_phr,1]
        if f_masks is not None:
            # alpha[:, 1:] = alpha[:, 1:].masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
            alpha = alpha.masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
        attw =  self.softmax(alpha) # [num_sen, num_phr, 1]
        # attw = F.tanh(alpha.transpose(1,2), dim=2)
        attw = attw.squeeze(-1)

        return attw


class DistilBert(nn.Cell):
    def __init__(self, joint_space_size, dataset, use_phrase, drop_phrase,magic_number=10):
        super().__init__()
        # for mindspore, use bert to replace distilbert
        # self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        bert_config = AutoConfig.from_pretrained('bert_base_uncased')
        self.bert = BertModel(bert_config)
        self.tokenizer = AutoTokenizer.from_pretrained('bert_base_uncased')

        self.fc_out1 = nn.Dense(768, joint_space_size)
        self.fc_out2 = nn.Dense(768, joint_space_size)
        self.dataset = dataset
        self.layernorm = nn.LayerNorm((768,))
        self.magic_number = magic_number
        self.aggregation = "avg"  # cls, avg
        self.use_phrase = use_phrase
        self.drop_phrase = drop_phrase
        if self.use_phrase:
            self.patt = AttentivePooling(joint_space_size, 128) # 128 is a magic number, remember to rewrite!
        # self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.joint_space_size = joint_space_size
    
    def encode_single(self, query, word_len):
        input_ids = query['input_ids']
        N, word_length = input_ids.shape[0], input_ids.shape[1]
        bert_encoding = self.bert(**query)[0] # [N, max_word_length, C]  .permute(2, 0, 1)
        if self.aggregation == "cls":
            query = bert_encoding[:, 0, :]  # [N, C], use [CLS] (first token) as the whole sentence feature
            query = self.layernorm(query)
            out_iou = self.fc_out1(query)
            out = self.fc_out2(query)
        elif self.aggregation == "avg":
            avg_mask = ops.zeros((N, word_length))
            for i in range(N):
                avg_mask[i, :word_len[i]] = 1       # including [CLS] (first token) and [SEP] (last token)
            avg_mask = avg_mask / (word_len.unsqueeze(-1))
            bert_encoding = bert_encoding.permute(2, 0, 1) * avg_mask  # use avg_pool as the whole sentence feature
            query = bert_encoding.sum(-1).t()  # [N, C]
            query = self.layernorm(query)
            out_iou = self.fc_out1(query)
            out = self.fc_out2(query)
        else:
            raise NotImplementedError
        return out, out_iou

    def construct(self, queries, wordlens):
        '''
        Average pooling over bert outputs among words to be sentence feature
        :param queries:
        :param wordlens:
        :param vid_avg_feat: B x C
        :return: list of [num_sent, C], len=Batch_size
        '''
        sent_feat = []
        sent_feat_iou = []
        for query, word_len in zip(queries, wordlens):  # each sample (several sentences) in a batch (of videos)
            out, out_iou = self.encode_single(query, word_len)
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        return sent_feat, sent_feat_iou
    
    
    def encode_sentences(self, sentences, phrases=None):
        sent_feat = []
        sent_feat_iou = []
        phrase_feat = []
        phrase_feat_iou = []
        phrase_weight = []
        phrase_masks = []

        stnc_query, stnc_len = bert_embedding_batch(sentences, self.tokenizer)
        for query, word_len in zip(stnc_query, stnc_len):  # each sample (several sentences) in a batch (of videos)
            out, out_iou = self.encode_single(query, word_len)
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        if self.use_phrase == True and phrases is not None:
            for bid, phrases_avid in enumerate(phrases):
                phrase_feat_avid = []
                phrase_feat_avid_iou = []
                # print('phrases_avid',phrases_avid)
                phrase_query, phrase_len = bert_embedding_batch(phrases_avid, self.tokenizer)
                # print('phrase_query',phrase_query)
                for query, word_len in zip(phrase_query, phrase_len):
                    query['input_ids'] = query['input_ids'][:self.magic_number]
                    query['input_mask'] = query['input_mask'][:self.magic_number]
                    query['token_type_ids'] = query['token_type_ids'][:self.magic_number]
                    # print('query',query['input_ids'].shape, query['input_mask'].shape, query['token_type_ids'].shape)
                    # print('word_len',word_len[:self.magic_number])
                    word_len = word_len[:self.magic_number]
                    # print(query)
                    # print('====================')
                    out, out_iou = self.encode_single(query, word_len)
                    pad_tensor = ops.zeros((self.magic_number-len(out), self.joint_space_size)) 
                    # print('pad_tensor',pad_tensor.shape)
                    out = ops.cat((out, pad_tensor), 0)
                    out_iou = ops.cat((out_iou, pad_tensor), 0)
                    phrase_feat_avid.append(out)
                    phrase_feat_avid_iou.append(out_iou)
                
                phrase_feat_avid = pad_sequence(phrase_feat_avid, batch_first=True)
                # print('phrase_feat_avid',phrase_feat_avid.shape)
                phrase_feat_avid_iou = pad_sequence(phrase_feat_avid_iou, batch_first=True)
                # print('phrase_feat_avid_iou',phrase_feat_avid_iou.shape)
                phrase_mask = ((phrase_feat_avid != 0).long().sum(axis=-1) != 0).long()
                # print('phrase_mask',phrase_mask.shape)
                

                
                if self.training and self.drop_phrase:
                    # print('drop phrase')
                    phrase_keep_weight = ops.zeros_like(phrase_mask).float()
                    for i, p in enumerate(phrases_avid):
                        for j, pp in enumerate(p[:10]):
                            phrase_keep_weight[i, j] = 0.9
                    drop_mask = ops.bernoulli(phrase_keep_weight)
                    phrase_mask = phrase_mask * drop_mask

                phrase_w = self.patt(phrase_feat_avid_iou, sent_feat_iou[bid], phrase_mask)
                
                phrase_feat.append(phrase_feat_avid)
                
                phrase_feat_iou.append(phrase_feat_avid_iou)
                
                phrase_weight.append(phrase_w)
                
                phrase_masks.append(phrase_mask)
            return sent_feat, sent_feat_iou, phrase_feat, phrase_feat_iou, phrase_weight, phrase_masks

        return sent_feat, sent_feat_iou


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    # type: (List[Tensor], bool, float) -> Tensor

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = ops.ones(out_dims) *padding_value 
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def bert_embedding_batch(sentences, tokenizer):
    queries = []
    word_lens = []
    for sentence in sentences:
        query_token = tokenizer(sentence,max_length=128, padding="max_length",return_tensors='ms')
        query_token['input_mask'] = query_token.pop('attention_mask')
        word_lens.append(query_token['input_mask'].sum(axis=1))
        queries.append(query_token)
    return queries, word_lens

def build_text_encoder(cfg):
    joint_space_size = cfg.MODEL.TRM.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME 
    use_phrase = cfg.MODEL.TRM.TEXT_ENCODER.USE_PHRASE
    drop_phrase = cfg.MODEL.TRM.TEXT_ENCODER.DROP_PHRASE
    return DistilBert(joint_space_size, dataset_name, use_phrase, drop_phrase)


if __name__ == '__main__':
    import numpy as np
    from mindspore import Tensor
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    attn_pool = AttentivePooling(512, 128)
    x = Tensor(np.random.rand(2,5, 512).astype(np.float32))
    y = Tensor(np.random.rand(2, 512).astype(np.float32))
    mask = Tensor(np.ones((2, 5)).astype(np.float32))
    print(attn_pool(x, y, mask).shape)
    
    bert = DistilBert(512, "activitynet", True, True)
    queries = [["hello world", "this is a test"],["hello world"]]
    phrases = [ [['hello','world'],['this','is','a','test']], [['hello','world']]]
    sent_feat, sent_feat_iou, phrase_feat, phrase_feat_iou, phrase_weight, phrase_masks = bert.encode_sentences(queries, phrases)
    print(sent_feat[0].shape, sent_feat_iou[0].shape, phrase_feat[0].shape, phrase_feat_iou[0].shape, phrase_weight[0].shape, phrase_masks[0].shape)
    