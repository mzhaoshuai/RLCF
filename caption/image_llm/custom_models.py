# coding=utf-8
import os
import math
import copy
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAP_TTA(nn.Module):
    def __init__(self, cap_model, device, momentum_update=False, update_freq=256, update_w=1.0, momentum=0.9998, cap_ckpt=None):
        """
        A wrapper for convinient operations
        """
        super().__init__()
        self.cap_model = cap_model
        print("Load model from {}...".format(cap_ckpt))
        self.cap_model.load_state_dict(torch.load(cap_ckpt, map_location=device)['state_dict'])
        print("Load model from {}...  Done".format(cap_ckpt))
        self.cap_model.to(device)

        self.device = device
        self.text_features = None
        self.image_features = None

        self.momentum_update = momentum_update
        self.update_freq = update_freq
        self.update_w = update_w
        self.momentum = momentum
        self.update_counter = 0
        # save model state of visual encoder
        with torch.no_grad():
            self.cap_state_dict = copy.deepcopy(self.cap_model.clip_project.state_dict())
            self.initial_state_dict = copy.deepcopy(self.cap_model.clip_project.state_dict())
            if self.momentum_update:
                self.momentum_state_dict = copy.deepcopy(self.cap_model.clip_project.state_dict())

        print("\n CAP_TTA model created: \n"
                "\t momentum_update / momentum / update_freq / update_w: [{} / {} / {} / {}] \n".format(
                    momentum_update, momentum, update_freq, self.update_w))

    @torch.no_grad()
    def reset_all(self):
        # reset the state dict of cap model, sometimes you may change the weights
        self.cap_model.clip_project.load_state_dict(self.cap_state_dict)
        self.initial_state_dict = copy.deepcopy(self.cap_model.clip_project.state_dict())
        if self.momentum_update:
            self.momentum_state_dict = copy.deepcopy(self.cap_model.clip_project.state_dict())

    @torch.no_grad()
    def reset_initial(self):
        self.cap_model.clip_project.load_state_dict(self.initial_state_dict)

    @torch.no_grad()
    def momentum_update_model(self):
        update_w = self.update_w
        if self.momentum_update:
            self.update_counter += 1
            # reload momentum state_dict
            state_dict = self.cap_model.clip_project.state_dict()
            for k, v in state_dict.items():
                self.momentum_state_dict[k]= self.momentum * self.momentum_state_dict[k] + (1.0 - self.momentum) * v

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                for k, v in state_dict.items():
                    # self.initial_state_dict[k] = self.momentum_state_dict[k]
                    self.initial_state_dict[k] = (1 - update_w) * self.cap_state_dict[k] + update_w * self.momentum_state_dict[k]
            # update will be done by function self.reset_initial()

    def parameters(self, recurse: bool = True):
        return self.cap_model.parameters()

    def train(self, mode: bool = True):
        return self.cap_model.train()

    @property
    def clip_project(self):
        return self.cap_model.clip_project

    @property
    def llm(self):
        return self.cap_model.llm

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        return self.cap_model(tokens, prefix, mask=mask, labels=labels)
