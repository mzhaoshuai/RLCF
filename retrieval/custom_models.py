# coding=utf-8
import os
import math
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lavis.models.clip_models.tokenizer import tokenize
from lavis.models.clip_models.model import load_openai_model


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


class CLIPRet_TTA(nn.Module):
    def __init__(self, device, arch="ViT-B-16", only_visual=True, momentum_update=False,
                    update_freq=256, update_w=1.0, momentum=0.9999):
        """
        Using CLIP to do image retrieval with Test Time Adaptation,
        this module will tune the whole or part parameters of CLIP image encoder or text encoder.
        Args:
            only_norm: if True, only update the normalization layer
        """
        super().__init__()
        model_path = os.path.join(DOWNLOAD_ROOT, arch + '.pt')
        self.clip_model = load_openai_model(model_path, device, jit=False)
        self.clip_model.float()

        self.device = device
        self.text_features = None
        self.image_features = None     
        # freeze parameters
        self.only_visual = only_visual
        self.freeze_parameters()

        self.momentum_update = momentum_update
        self.update_freq = update_freq
        self.update_w = update_w
        self.momentum = momentum
        self.update_counter = 0
        # save model state of visual encoder
        with torch.no_grad():
            self.clip_state_dict = copy.deepcopy(self.clip_model.state_dict())
            self.initial_state_dict = copy.deepcopy(self.clip_model.state_dict())
            if self.momentum_update:
                self.momentum_state_dict = copy.deepcopy(self.clip_model.state_dict())

        print("\n CLIPRet_TTA model created: \n"
                "\t backbone: {}, momentum_update / momentum / update_freq / update_w: [{} / {} / {} / {}] \n".format(
                    arch, momentum_update, momentum, update_freq, self.update_w))

    def forward(self, images=None, text=None, tokenized_prompts=None):
        image_features = self.get_image_features(images) if images is not None else self.image_features
        text_features = self.get_text_features(text, tokenized_prompts) if text is not None or tokenized_prompts is not None \
                        else self.text_features

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text

    def get_text_features(self, text=None, tokenized_prompts=None):
        if tokenized_prompts is None:
            assert text is not None
            tokenized_prompts = tokenize(text).to(self.device)

        text_features = self.clip_model.encode_text(tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def get_image_features(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def set_image_features(self, images=None, image_features=None):
        if images is not None:
            self.image_features = self.get_image_features(images)
        else:
            self.image_features = image_features

    def set_text_features(self, text=None, tokenized_prompts=None, text_features=None):
        if text is not None or tokenized_prompts is not None:
            self.text_features = self.get_text_features(text, tokenized_prompts)
        else:
            assert text_features is not None
            self.text_features = text_features

    @torch.no_grad()
    def freeze_parameters(self):
        if self.only_visual:
            for n, p in self.clip_model.named_parameters():
                if 'visual' not in n:
                    p.requires_grad_(False)
            self.clip_model.transformer.eval()
            self.clip_model.ln_final.eval()
        else:
            self.clip_model.lock_image_tower(unlocked_groups=0, freeze_bn_stats=True)
            self.clip_model.visual.eval()

    @torch.no_grad()
    def reset_all(self):
        # reset the state dict of clip model, sometimes you may change the weights
        self.clip_model.load_state_dict(self.clip_state_dict)
        self.initial_state_dict = copy.deepcopy(self.clip_model.state_dict())
        if self.momentum_update:
            self.momentum_state_dict = copy.deepcopy(self.clip_model.state_dict())

    @torch.no_grad()
    def reset_initial(self):
        self.clip_model.load_state_dict(self.initial_state_dict)

    @torch.no_grad()
    def momentum_update_model(self):
        update_w = self.update_w
        if self.momentum_update:
            self.update_counter += 1
            # reload momentum state_dict
            state_dict = self.clip_model.state_dict()
            for k, v in state_dict.items():
                self.momentum_state_dict[k]= self.momentum * self.momentum_state_dict[k] + (1.0 - self.momentum) * v

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                for k, v in state_dict.items():
                    # self.initial_state_dict[k] = self.momentum_state_dict[k]
                    self.initial_state_dict[k] = (1 - update_w) * self.clip_state_dict[k] + update_w * self.momentum_state_dict[k]
            # update will be done by function self.reset_initial()

    def parameters(self, recurse: bool = True):
        if self.only_visual:
            return self.clip_model.visual.parameters()
        else:
            params = []
            for n, p in self.clip_model.named_parameters():
                if 'visual' not in n:
                    params.append(p)
            return params

    def train(self, mode: bool = True):
        super().train(mode)

        if self.only_visual:
            self.clip_model.transformer.eval()
            self.clip_model.ln_final.eval()
        else:
            self.clip_model.visual.eval()

        return self


class CLIPRet_Multiple(nn.Module):
    def __init__(self, device, arch=["ViT-B-16", "ViT-L-14"], **kwargs):
        """
        For CLIP ensemble, zero-shot
        """
        super().__init__()
        # load clip models
        clip_models = []
        self.resolutions = []
        self.default_resolutions = kwargs["default_resolutions"] if "default_resolutions" in kwargs.keys() else 224
        for ar in arch:
            model_path = os.path.join(DOWNLOAD_ROOT, ar + '.pt')
            clip_model = load_openai_model(model_path, device, jit=False)
            clip_model.float()
            clip_models.append(clip_model)
            self.resolutions.append(clip_model.visual.image_size)
        self.clip_models = nn.ModuleList(clip_models)
        self.n_model = len(self.clip_models)
        self.device = device
        self.text_features = None
        self.image_features = None     

        print("\n CLIPRet_TTA model created: backbone: {} \n".format(arch))

    def forward(self, images=None, text=None, tokenized_prompts=None):
        sim_matrix_i2t_list, sim_matrix_t2i_list = [], []
        for i in range(self.n_model):
            image_features = self.image_features[i]
            text_features = self.text_features[i]
            sims_matrix_i2t = image_features @ text_features.t()
            sims_matrix_t2i = sims_matrix_i2t.t()
            sim_matrix_i2t_list.append(sims_matrix_i2t)
            sim_matrix_t2i_list.append(sims_matrix_t2i)

        sims_matrix_i2t = torch.mean(torch.stack(sim_matrix_i2t_list, dim=0), dim=0)
        sims_matrix_t2i = torch.mean(torch.stack(sim_matrix_t2i_list, dim=0), dim=0)

        return sims_matrix_i2t, sims_matrix_t2i

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

        # list of tensor with len=n_model and shape [batch_size, dim]
        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        text_features = []
        for i in range(self.n_model):
            if captions is not None:
                caption_tokens = tokenize(captions).to(self.device)
                text_feat = self.clip_models[i].encode_text(caption_tokens).float()

            if tokenized_cap is not None:
                text_feat = self.clip_models[i].encode_text(tokenized_cap).float()

            # normalized features
            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            text_features.append(text_feat)

        # list of tensor with len=n_model and shape [batch_size, dim]
        return text_features

    @torch.no_grad()
    def set_many_text_features(self, texts, text_bs=128):
        """tokenize a list of 'texts' and call self.set_class_features()"""
        num_text = len(texts)
        text_feats = [[] for _ in range(self.n_model)]
        i = 0
        while i < num_text:
            text = texts[i : min(num_text, i + text_bs)]
            input_ids = tokenize(text).to(self.device)
            text_features_list = self.extract_text_features(tokenized_cap=input_ids)
            for j in range(self.n_model):
                text_feats[j].append(text_features_list[j])
            i += text_bs

        text_feats_tensors = []
        for i in range(self.n_model):
            tensor = torch.cat(text_feats[i], dim=0)
            text_feats_tensors.append(tensor)

        self.text_features = text_feats_tensors

    @torch.no_grad()
    def set_image_features_with_dataloder(self, data_loader):
        image_feats = [[] for _ in range(self.n_model)]
        for samples in data_loader:
            image = samples["image"].to(self.device)
            image_features_list = self.extract_image_features(image)
            for j in range(self.n_model):
                image_feats[j].append(image_features_list[j])

        image_feats_tensors = []
        for i in range(self.n_model):
            tensor = torch.cat(image_feats[i], dim=0)
            image_feats_tensors.append(tensor)

        self.image_features = image_feats_tensors
