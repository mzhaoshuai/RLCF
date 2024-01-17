# coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Tuple, Optional
from transformers import GPT2LMHeadModel

from image_llm.models.modeling_opt import OPTForCausalLM


class MLP(nn.Module):

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=F.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)

        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=F.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=F.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))

        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x


class TransformerMapper(nn.Module):

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, clip_patch = False):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.clip_patch = clip_patch

        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, dim_embedding if clip_patch else clip_length * dim_embedding)

        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)

        # prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        # out = self.transformer(x)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]

        return out


class TransformerEncoderDecoder(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        ref = self.ref_encoder(x)
        const = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = self.prefix_decoder(const, ref)
        return prefix

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 4):
        super(TransformerEncoderDecoder, self).__init__()
        self.clip_length = clip_length
        self.ref_encoder = Transformer(512, 8, num_layers)
        self.prefix_decoder = Transformer(dim_embedding, 8, num_layers, dim_ref=512, enc_dec=True)
        self.linear = nn.Linear(dim_clip, clip_length * 512)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'
    TransformerEncoder = 'transformer_encoder'
    TransformerDecoder = 'transformer_decoder'


class LLMModel(nn.Module):
    """large language model"""
    def __init__(self, config_dir):
        super().__init__()
        self.model_name = os.path.basename(config_dir)

        if "gpt2" in config_dir:
            self.llm = GPT2LMHeadModel.from_pretrained(config_dir)
        elif "opt" in config_dir:
            self.llm = OPTForCausalLM.from_pretrained(config_dir, torch_dtype=torch.float16)
        else:
            raise NotImplementedError("{} not found".format(config_dir))

    def forward(self, inputs_embeds=None, labels=None, attention_mask=None):
        return self.llm(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)

    @property
    def embedding_size(self):
        return self.llm.get_input_embeddings().weight.shape[1]

    def input_token_embedding(self):
        return self.llm.get_input_embeddings()


class ClipCaptionModelV2(nn.Module):
    """version 2 of ClipCaption Model, provide alternative choice of LLM models"""
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                    num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, config_dir: str = "gpt2",
                    clip_patch=False):
        super(ClipCaptionModelV2, self).__init__()
        self.prefix_length = prefix_length
        self.clip_patch = clip_patch

        self.llm = LLMModel(config_dir)
        self.llm_embedding_size = self.llm.embedding_size

        if mapping_type == MappingType.MLP:
            if not self.clip_patch:
                sizes = (prefix_size, (self.llm_embedding_size * prefix_length) // 2, self.llm_embedding_size * prefix_length)
            else:
                sizes = (prefix_size, self.llm_embedding_size // 2, self.llm_embedding_size)
            
            self.clip_project = MLP(sizes)
        else:
            self.clip_project = TransformerMapper(prefix_size, self.llm_embedding_size, prefix_length,
                                                    clip_length, num_layers,
                                                    clip_patch=self.clip_patch)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.llm.input_token_embedding()(tokens)

        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.llm_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.llm(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)

        return out


class ClipCaptionPrefixV2(ClipCaptionModelV2):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefixV2, self).train(mode)

        for n, p in self.llm.named_parameters():
            p.requres_grad = False

        self.llm.eval()

        return self


# original ClipCap code
# class ClipCaptionModel(nn.Module):

#     def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
#         return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

#     def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
#                 labels: Optional[torch.Tensor] = None):
#         embedding_text = self.gpt.transformer.wte(tokens)
#         prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
#         embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
#         if labels is not None:
#             dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
#             labels = torch.cat((dummy_token, tokens), dim=1)
#         out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
#         return out

#     def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
#                  num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
#         super(ClipCaptionModel, self).__init__()
#         self.prefix_length = prefix_length
#         self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
#         if mapping_type == MappingType.MLP:
#             self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
#                                      self.gpt_embedding_size * prefix_length))
#         else:
#             self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
#                                                                      clip_length, num_layers)


# class ClipCaptionPrefix(ClipCaptionModel):

#     def parameters(self, recurse: bool = True):
#         return self.clip_project.parameters()

#     def train(self, mode: bool = True):
#         super(ClipCaptionPrefix, self).train(mode)
#         self.gpt.eval()
#         return self
