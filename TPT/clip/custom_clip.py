# coding=utf-8
import os
import math
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

_tokenizer = _Tokenizer()

### This is my deafult PATH
DOWNLOAD_ROOT_v1 = '/home/shuzhao/Data/pretrained/clip'
### PUT YOUR PATH HERE
DOWNLOAD_ROOT_v2 = '/YOUR/PATH'

if os.path.exists(DOWNLOAD_ROOT_v1):
    DOWNLOAD_ROOT = DOWNLOAD_ROOT_v1
elif os.path.exists(DOWNLOAD_ROOT_v2):
    DOWNLOAD_ROOT = DOWNLOAD_ROOT_v2
else:
    DOWNLOAD_ROOT='~/.cache/clip'
    raise NotImplementedError("You shoud put an available CLIP download folder here")


class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        """reset the prompt to be the initial state"""
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) 
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model


class CLIPCLS_TTA(nn.Module):
    def __init__(self, device, classnames, arch="ViT-L/14", prompt_prefix=None, only_visual=True, momentum_update=False,
                    update_freq=256, update_w=1.0, momentum=0.9999, only_norm=False):
        """
        Using CLIP to do classification with Test Time Adaptation, different from Class 'ClipTestTimeTuning',
        this module will tune the whole or part parameters of CLIP image encoder or text encoder.
        Args:
            only_norm: if True, only update the normalization layer
        """
        super().__init__()
        self.clip_model, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.device = device
        # get classnames and features
        self.prompt_prefix = prompt_prefix
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(classnames)
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        self.tokenized_prompts = tokenize(prompts).to(self.device)
        self.class_features = self.get_class_features(self.tokenized_prompts)
        
        # freeze parameters
        self.only_visual = only_visual
        self.only_norm = only_norm
        self.freeze_parameters()

        self.momentum_update = momentum_update
        self.update_freq = update_freq
        self.update_w = update_w
        self.momentum = momentum
        self.update_counter = 0
        # save model state of visual encoder
        with torch.no_grad():
            self.clip_state_dict = copy.deepcopy(self.clip_model.visual.state_dict())
            self.initial_state_dict = copy.deepcopy(self.clip_model.visual.state_dict())
            if self.momentum_update:
                self.momentum_state_dict = copy.deepcopy(self.clip_model.visual.state_dict())

        print("\n CLIPCLS_TTA model created: \n"
                "\t backbone: {}, only_norm: {}, momentum_update / momentum / update_freq / update_w: [{} / {} / {} / {}] \n".format(
                    arch, only_norm, momentum_update, momentum, update_freq, self.update_w))

    @torch.no_grad()
    def get_class_features(self, tokenized_prompts):
        class_features = self.clip_model.encode_text(tokenized_prompts)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        return class_features

    @torch.no_grad()
    def freeze_parameters(self):
        if self.only_visual:
            for n, p in self.clip_model.named_parameters():
                if 'visual' not in n:
                    p.requires_grad_(False)
                # else:
                #     if self.only_norm and ('ln' not in n or 'bn' not in n):
                #         p.requires_grad_(False)
        self.clip_model.transformer.eval()
        self.clip_model.ln_final.eval()

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.class_features

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    @torch.no_grad()
    def reset_classnames_and_state(self, classnames, arch):
        self.n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        self.classnames = classnames
        self.tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        if not self.only_visual:
            clip_m, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)
            class_features = clip_m.encode_text(self.tokenized_prompts)
            self.class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        else:
            self.class_features = self.get_class_features(self.tokenized_prompts)

        # reset the state dict of clip model, sometimes you may change the weights
        self.clip_model.visual.load_state_dict(self.clip_state_dict)
        self.initial_state_dict = copy.deepcopy(self.clip_model.visual.state_dict())
        if self.momentum_update:
            self.momentum_state_dict = copy.deepcopy(self.clip_model.visual.state_dict())

    @torch.no_grad()
    def reset(self):
        self.clip_model.visual.load_state_dict(self.initial_state_dict)

    @torch.no_grad()
    def momentum_update_model(self):
        update_w = self.update_w
        if self.momentum_update:
            self.update_counter += 1
            # reload momentum state_dict
            state_dict = self.clip_model.visual.state_dict()
            for k, v in state_dict.items():
                self.momentum_state_dict[k]= self.momentum * self.momentum_state_dict[k] + (1.0 - self.momentum) * v

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                for k, v in state_dict.items():
                    # self.initial_state_dict[k] = self.momentum_state_dict[k]
                    self.initial_state_dict[k] = (1 - update_w) * self.clip_state_dict[k] + update_w * self.momentum_state_dict[k]
            # update will be done by function self.reset()

    def parameters(self, recurse: bool = True):
        if not self.only_norm:
            return self.clip_model.visual.parameters()
        else:
            params = []
            for n, p in self.clip_model.visual.named_parameters():
                if 'ln' in n or 'bn' in n:
                    params.append(p)
            return params

    def train(self, mode: bool = True):
        super().train(mode)
        if self.only_norm:
            for n, m in self.clip_model.visual.named_modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                    m.train()
                else:
                    m.eval()
        self.clip_model.transformer.eval()
        self.clip_model.ln_final.eval()
        return self


class CLIPCLS_TTA_Multiple(nn.Module):
    def __init__(self, device, classnames, arch=["ViT-B/16", "ViT-L/14"], prompt_prefix=None, **kwargs):
        """
        For model ensemble test. It is not used for training in this project.
        """
        super().__init__()
        clip_models = []
        self.resolutions = []
        for ar in arch:
            clip_model, _, _ = load(ar, device=device, download_root=DOWNLOAD_ROOT)
            clip_models.append(clip_model)
            self.resolutions.append(clip_model.visual.input_resolution)
        self.clip_models = nn.ModuleList(clip_models)
        self.n_model = len(self.clip_models)
        self.default_resolutions = kwargs["default_resolutions"] if "default_resolutions" in kwargs.keys() else 224

        self.device = device
        # get classnames and features
        self.prompt_prefix = prompt_prefix
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(classnames)
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        self.tokenized_prompts = tokenize(prompts).to(self.device)
        self.class_features = self.get_class_features(self.tokenized_prompts)

    @torch.no_grad()
    def get_class_features(self, tokenized_prompts):
        text_features = []
        for i in range(self.n_model):
            text_feat = self.clip_models[i].encode_text(tokenized_prompts).float()
            # normalized features
            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            text_features.append(text_feat)

        return text_features

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features without normalization"""
        image_features = []
        for i in range(self.n_model):
            if self.resolutions[i] != self.default_resolutions:
                tmp_images = nn.functional.interpolate(images, size=self.resolutions[i], mode='bicubic', align_corners=True)
            else:
                tmp_images = images
            image_feat = self.clip_models[i].encode_image(tmp_images).float()
            image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
            image_features.append(image_feat)

        return image_features

    @torch.no_grad()
    def freeze_parameters(self):
        pass

    def forward(self, image):
        all_logits = []
        image_features = self.extract_image_features(image)
        text_features = self.class_features

        for i in range(self.n_model):
            logit_scale = self.clip_models[i].logit_scale.exp()
            logits = logit_scale * image_features[i] @ text_features[i].t()
            all_logits.append(logits)

        logits = torch.mean(torch.stack(all_logits, dim=0), dim=0)
        return logits

    @torch.no_grad()
    def reset_classnames_and_state(self, classnames, arch):
        self.n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        self.classnames = classnames
        self.tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        self.class_features = self.get_class_features(self.tokenized_prompts)

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def momentum_update_model(self):
        pass
