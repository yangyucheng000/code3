from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
import mindspore.nn as nn
from mindspore import ops
from mindspore import Tensor
import math
import pdb
from typing import Any, Union, List
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from pkg_resources import packaging
from typing import Optional, List
from transformer_module import TransformerDecoderLayer, TransformerDecoderLayer_womhsa
import random

# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoderLayer(nn.Cell):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(in_channels=d_model, out_channels=dim_feedforward)
        self.dropout = nn.Dropout(p=1.0 - dropout)
        self.linear2 = nn.Dense(in_channels=dim_feedforward, out_channels=d_model)

        self.norm1 = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
        self.norm2 = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
        self.norm3 = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
        self.dropout1 = nn.Dropout(p=1.0 - dropout)
        self.dropout2 = nn.Dropout(p=1.0 - dropout)
        self.dropout3 = nn.Dropout(p=1.0 - dropout)

        # self.activation = _get_activation_fn(activation)
        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2, attentions = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        # tgt2, attentions = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt2 = self.norm2(tgt)
        tgt2, attentions = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def construct(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


import copy
def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


class Adapter(nn.Cell):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.1,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_num_layers=1,
                 ):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        if adapter_scalar == "learnable_scalar":
            self.scale = ms.Parameter(ops.ones(d_model)*1e-9)
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Dense(in_channels=self.n_embd, out_channels=self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Dense(in_channels=self.down_size, out_channels=self.n_embd)
        self.adapter_num_layers = adapter_num_layers

        self.dropout = dropout
        # if init_option == "bert":
        #     raise NotImplementedError
        # elif init_option == "lora":
        #     with torch.no_grad():
        #         nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        #         nn.init.zeros_(self.up_proj.weight)
        #         nn.init.zeros_(self.down_proj.bias)
        #         nn.init.zeros_(self.up_proj.bias)
        
        instance_decoder_layer = TransformerDecoderLayer(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)
        instance_decoder_norm = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
        self.mhsa_layers = _get_clones(instance_decoder_layer, adapter_num_layers)
        # self.mhsa = TransformerDecoderLayer(self.down_size, 2, self.down_size*2,
        #                                         self.dropout, 'relu', False)

    def construct(self, x, prior=None, prior_mask=None):
        down = self.down_proj(x)
        down = self.non_linear_func(down) ## 197 x batchsize x 64
        if prior is not None:
            context, mask = prior, prior_mask
            context = context.swapaxes(0,1) ## 18(#instance) x batchsize x 64
            for z, layer in enumerate(self.mhsa_layers):
                down = layer(down, context, tgt_mask=None,
                            memory_mask=None,
                            tgt_key_padding_mask=None,
                            memory_key_padding_mask=mask,
                            pos=None, query_pos=None)
        else:
            down = self.mhsa.forward_post(down, down, tgt_mask=None,
                           memory_mask=None,
                           tgt_key_padding_mask=None,
                           memory_key_padding_mask=None,
                           pos=None, query_pos=None)
        up = self.up_proj(down)
        output = up * self.scale
        return output


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)


class QuickGELU(nn.Cell):
    def construct(self, x):
        return x * ops.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None, adapter: bool=False, adapter_num_layers: int=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
        self.mlp = nn.SequentialCell(OrderedDict([
            ("c_fc", nn.Dense(in_channels=d_model, out_channels=d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Dense(in_channels=d_model * 4, out_channels=d_model))
        ]))
        self.ln_2 = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
        self.attn_mask = attn_mask
        if adapter:
            self.adaptermlp = Adapter(None,  d_model=d_model , dropout=0.1, bottleneck=64,
                                    init_option='lora',
                                    adapter_scalar='learnable_scalar',
                                    adapter_num_layers=adapter_num_layers,
                                    ) 
        self.adapter = adapter

    def attention(self, x: Tensor):
        if self.attn_mask is not None:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask.to(x.dtype))[0]
        else:
            return self.attn(x, x, x, need_weights=False)[0]
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def construct(self, x, prior=None, prior_mask=None):
        '''
        x: L * bs * C, 
        prior: bs * L' * C', padded prior knowledge
        prior_mask: bs * L' (mask of prior knowledge)
        '''
        # x, prior = x  
        if self.adapter:
            adapt_x = self.adaptermlp(x, prior=prior, prior_mask=prior_mask)
            x = x + adapt_x
        x = x + self.attention(self.ln_1(x))  
        x = x + self.mlp(self.ln_2(x))            
        return x, prior, prior_mask


class Transformer(nn.Cell):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None, adapter: bool=False, adapter_layers: List=[i for i in range(24)], adapter_num_layers: int=1):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell([*[ResidualAttentionBlock(width, heads, attn_mask, adapter=((_ in adapter_layers) and adapter), adapter_num_layers=adapter_num_layers) for _ in range(layers)]])

    def construct(self, x: Tensor, prior: Tensor = None, prior_mask: Tensor = None):
        for resblock in self.resblocks:
            # if prior is not None:
            #     import pdb; pdb.set_trace()
            x, prior, prior_mask = resblock(x, prior, prior_mask)
        return x
        # return self.resblocks(x, prior, prior_mask)[0]


class VisionTransformer(nn.Cell):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, use_adapter: bool=True, adapter_layers: List=[_ for _ in range(24)], adapter_num_layers: int=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, pad_mode='pad', has_bias=False)

        scale = width ** -0.5
        self.class_embedding = ms.Parameter(scale * ops.randn(width))
        self.positional_embedding = ms.Parameter(scale * ops.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(normalized_shape=[width], epsilon=1e-05)

        self.transformer = Transformer(width, layers, heads, adapter=use_adapter, adapter_layers=adapter_layers, adapter_num_layers=adapter_num_layers)

        self.ln_post = nn.LayerNorm(normalized_shape=[width], epsilon=1e-05)
        self.proj = ms.Parameter(scale * ops.randn(width, output_dim))
        self.patch_size = patch_size

    def construct(self, x: Tensor, prior=None, prior_mask=None):
        bs, c, h, w = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = P.Reshape()(x, (x.shape[0], x.shape[1], -1,))  # shape = [*, width, grid ** 2]
        x = P.Transpose()(x, (0, 2, 1,))  # shape = [*, grid ** 2, width]
        x = P.Concat(1)([self.class_embedding.to(x.dtype) + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x])  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = P.Transpose()(x, (1, 0, 2,))  # NLD -> LND
        x = self.transformer(x, prior, prior_mask)
        x = P.Transpose()(x, (1, 0, 2,))  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x[:,0,:], x[:,1:,:].view(bs,h//self.patch_size, w//self.patch_size, -1).permute(0, 3, 1, 2)

    # def init_weights(self, pretrained=None):
    #     pretrained = pretrained or self.pretrained
        
    #     if isinstance(pretrained, str):
    #         checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
    #         state_dict = {}
    #         for k in checkpoint.keys():
    #             if not k.startswith('visual') and not k.startswith('transf'):
    #                 print(k)

    #         for k in checkpoint.keys():
    #             if k.startswith('visual.'):
    #                 new_k = k.replace('visual.', '')
    #                 state_dict[new_k] = checkpoint[k]

    #                 if 'positional_embedding' in new_k:
    #                     if self.positional_embedding.shape != state_dict[new_k].shape:
    #                         print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.positional_embedding.shape}')
    #                         cls_pos = state_dict[new_k][0:1, :]
    #                         H = W = self.input_resolution // 16
    #                         spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 14, 14, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
    #                         # H = W = self.input_resolution // 32
    #                         # spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 7, 7, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
    #                         spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
    #                         positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
    #                         state_dict[new_k] = positional_embedding
    #                         print(self.positional_embedding.shape , state_dict[new_k].shape,self.input_resolution)
    #                         assert self.positional_embedding.shape == state_dict[new_k].shape
    #         
    #         u, w = self.load_state_dict(state_dict, False)
    #         u = [k for k in u if not 'adaptermlp' in k]
    #         print(u, w, 'are misaligned params in CLIPResNet')


class CLIP(nn.Cell):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 use_adapter=True,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        # if isinstance(vision_layers, (tuple, list)):
        #     vision_heads = vision_width * 32 // 64
        #     self.visual = ModifiedResNet(
        #         layers=vision_layers,
        #         output_dim=embed_dim,
        #         heads=vision_heads,
        #         input_resolution=image_resolution,
        #         width=vision_width
        #     )
        # else:
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            use_adapter=use_adapter,
            adapter_layers=kwargs["adapter_layers"],
            adapter_num_layers=kwargs["adapter_num_layers"],
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = ms.Parameter(ms.numpy.empty((self.context_length, transformer_width)))
        self.ln_final = nn.LayerNorm(normalized_shape=[transformer_width], epsilon=1e-05)

        self.text_projection = ms.Parameter(ms.numpy.empty((transformer_width, embed_dim)))
        self.logit_scale = ms.Parameter(ops.ones([]) * np.log(1 / 0.07))

        # self.initialize_parameters()

    # def initialize_parameters(self):
    #     nn.init.normal_(self.token_embedding.weight, std=0.02)
    #     nn.init.normal_(self.positional_embedding, std=0.01)

    #     # if isinstance(self.visual, ModifiedResNet):
    #     #     if self.visual.attnpool is not None:
    #     #         std = self.visual.attnpool.c_proj.in_features ** -0.5
    #     #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
    #     #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
    #     #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
    #     #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

    #     #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
    #     #         for name, param in resnet_block.named_parameters():
    #     #             if name.endswith("bn3.weight"):
    #     #                 nn.init.zeros_(param)

    #     proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
    #     attn_std = self.transformer.width ** -0.5
    #     fc_std = (2 * self.transformer.width) ** -0.5
    #     for block in self.transformer.resblocks:
    #         nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
    #         nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
    #         nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
    #         nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    #     if self.text_projection is not None:
    #         nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        mask = ops.fill(ms.float32, (self.context_length, self.context_length), float("-inf"))
        mask = mask.triu(1)  # zero out the lower diagonal
        return mask
        # # pytorch uses additive attention mask; fill with -inf
        # mask = torch.empty(self.context_length, self.context_length)
        # mask.fill_(float("-inf"))
        # mask.triu_(1)  # zero out the lower diagonal
        # return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[ops.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection

        return x

    def construct(self, image, text):
        return 0


# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""

#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()

#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

#         for name in ["text_projection", "proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 if attr is not None:
#                     attr.data = attr.data.half()

#     model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, use_adapter=True, adapter_pos='all', adapter_num_layers=1):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.embedding_table"].shape[0]
    transformer_width = state_dict["ln_final.gamma"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    if adapter_pos == 'all':
        adapter_layers = [z for z in range(vision_layers)]
    elif adapter_pos == 'front':
        adapter_layers = [z for z in range(vision_layers // 2)]
    elif adapter_pos == 'end':
        adapter_layers = [z for z in range(vision_layers//2, vision_layers)]
    elif adapter_pos == 'last':
        adapter_layers = [z for z in range(vision_layers-1, vision_layers)]
    elif adapter_pos == 'random':
        adapter_layers = [random.randint(0, vision_layers-1) for z in range(vision_layers//2)] 

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, use_adapter=use_adapter, 
        adapter_layers=adapter_layers, adapter_num_layers=adapter_num_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # print('[INFO] missing_keys:', [ k for k in missing_keys if 'adaptermlp' not in k])
    # print('[INFO] unexpected_keys:', unexpected_keys)
    ms.load_param_into_net(model, state_dict)
    return model.set_train(False)
