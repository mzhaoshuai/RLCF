# coding=utf-8
import os
import torch
import torch.nn as nn

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


CONFIDECES = {
        "ViT-L-14-336": 10,
        "ViT-L-14-336px": 10,
        "ViT-L-14": 5,
        "RN50x64": 3,
        "ViT-B-16": 1
    }


def get_reward_model(device, args):
    if args.multiple_reward_models:
        reward_model = CLIPRewardsMultiple(device, arch=["ViT-L-14-336px", "RN50x64", "ViT-L-14"], classification=True,
                            amplify_rewards=args.reward_amplify, sample_k=args.sample_k,
                            reward_process=args.reward_process, process_batch=args.process_batch,
                            weighted_scores=args.weighted_scores)

    else:
        reward_model = CLIPRewards(device, arch=args.reward_arch, classification=True,
                                amplify_rewards=args.reward_amplify, sample_k=args.sample_k,
                                reward_process=args.reward_process, process_batch=args.process_batch)

    return reward_model


class BaseRewards(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_features = None
        self.image_features = None
        self.reward_process = True
        self.amplify_rewards = False

    @torch.no_grad()
    def extract_image_features(self, images):
        pass

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        pass

    @torch.no_grad()
    def set_text_features(self, captions=None, tokenized_cap=None, text_features=None):
        if text_features is None:
            self.text_features = self.extract_text_features(captions=captions, tokenized_cap=tokenized_cap)
        else:
            self.text_features = text_features

    @torch.no_grad()
    def set_image_features(self, images=None, image_features=None):
        if image_features is None:
            assert images is not None
            self.image_features = self.extract_image_features(images)
        else:
            self.image_features = image_features

    @torch.no_grad()
    def confidence_gap(self, predictions):
        """
        Args:
            predictions: shape [bs, C]
        """
        value, index = torch.topk(predictions, 2, dim=-1)
        gap = value[:, 0] - value[:, 1]
        gap = gap - torch.mean(gap)

        return gap

    @torch.no_grad()
    def rewards_post_process(self, clip_score):
        """
        clip_score: shape [bs, K] or [bs * K]
        """
        # if (clip_score.ndim > 1 and clip_score.shape[-1] > 1) or (clip_score.ndim == 1 and clip_score.shape[-1] > 1):
        if clip_score.shape[-1] > 1 and self.reward_process:
            mean = torch.mean(clip_score, dim=-1, keepdim=True)
            if self.amplify_rewards:
                std = torch.std(clip_score, dim=-1, keepdim=True) + 1e-5
            else:
                std = 1.0
            clip_score = (clip_score - mean) / std

        return clip_score.flatten()


class CLIPRewards(BaseRewards):
    def __init__(self, device, arch="ViT-B-16", clipscore_weight=2.5, classification=True,
                    amplify_rewards=False, sample_k=5, reward_process=True, process_batch=False,
                    default_resolutions=224) -> None:
        """
        calculating CLIP Reward
        Args:
            clipscore_weight: weight for calculating CLIPScore
            reward_process: If ture, post-process the rewards, e.g., subtract the reward mean or standardization
            amplify_rewards: If true, after subtracting the reward mean, also divide rewards by standard variance of rewards, i.e, standardization.
            sample_k: K for sampling.
        """
        super().__init__()
        self.default_resolutions = default_resolutions
        # self.clip_model, self.embed_dim, self.preprocess = clip.load(arch, device=device, download_root=DOWNLOAD_ROOT)
        model_path = os.path.join(DOWNLOAD_ROOT, arch + '.pt')
        self.clip_model = load_openai_model(model_path, device, jit=False)
        self.clip_model.float()

        self.resolutions = self.clip_model.visual.image_size
        self.clipscore_weight = clipscore_weight
        self.device = device
        self.classification = classification
        self.text_features = None
        self.image_features = None
        self.amplify_rewards = amplify_rewards
        self.sample_k = sample_k
        self.reward_process = reward_process
        self.process_batch = process_batch
        self.clip_model.eval()

        print("\n CLIPRewards model created: \n"
                "\t visual backbone: {}, resolutions: {}, amplify_rewards: {}, sample_k: {}, \n"
                "\t reward_process: {}, process_batch: {}\n".format(
                    arch, self.resolutions, amplify_rewards, sample_k, reward_process, process_batch))

    @torch.no_grad()
    def CLIPScore(self, text_index=None, images_index=None, pairwise=True):
        """
        class_index: sampled class index
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # suitable for image-2-text retrieval
        if text_index is not None:
            text_features = self.text_features[text_index]
        else:
            text_features = torch.repeat_interleave(self.text_features, self.sample_k, dim=0)

        # suitable for text-2-image retrieval
        if images_index is not None:
            image_features = self.image_features[images_index]
        else:
            image_features = torch.repeat_interleave(self.image_features, self.sample_k, dim=0)

        if pairwise:
            similarity = self.clipscore_weight * text_features @ image_features.t()
        else:
            similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

        scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()

        return scores

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features with normalization"""
        if self.resolutions != self.default_resolutions:
            images = nn.functional.interpolate(images, size=self.resolutions, mode='bicubic', align_corners=True)
        image_features = self.clip_model.encode_image(images).float()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        if captions is not None:
            caption_tokens = tokenize(captions).to(self.device)
            text_features = self.clip_model.encode_text(caption_tokens).float()
        if tokenized_cap is not None:
            text_features = self.clip_model.encode_text(tokenized_cap).float()
        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    @torch.no_grad()
    def set_many_text_features(self, texts, text_bs=128):
        """tokenize a list of 'texts' and call self.set_class_features()"""
        num_text = len(texts)
        text_feats = []
        i = 0
        while i < num_text:
            text = texts[i : min(num_text, i + text_bs)]
            input_ids = tokenize(text).to(self.device)
            text_features = self.extract_text_features(tokenized_cap=input_ids)
            text_feats.append(text_features)
            i += text_bs

        self.text_features = torch.cat(text_feats, dim=0)

    @torch.no_grad()
    def set_image_features_with_dataloder(self, data_loader):
        image_feats = []
        for samples in data_loader:
            image = samples["image"].to(self.device)
            image_features = self.extract_image_features(image)
            image_feats.append(image_features)

        self.image_features = torch.cat(image_feats, dim=0)

    @torch.no_grad()
    def calulate_similarity(self):
        """
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * self.image_features @ self.text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class CLIPRewardsMultiple(BaseRewards):
    def __init__(self, device, arch=["ViT-B/16", "RN50x64", "ViT-L/14"], clipscore_weight=2.5, classification=False,
                    amplify_rewards=False, sample_k=5, reward_process=True, process_batch=True, weighted_scores=True,
                    default_resolutions=224) -> None:
        """calculating CLIP Reward using multiple CLIP models
        Args:
            arch: a list of CLIP arches
            clipscore_weight: weight for calculating CLIPScore
            weighted_scores: if true, the final score is the weighted average of all scores
        """
        super().__init__()
        # load clip models
        clip_models = []
        # self.preprocess = []
        self.resolutions = []
        weights = []
        self.default_resolutions = default_resolutions
        for ar in arch:
            # clip_model, embed_dim, preprocess = clip.load(ar, device=device, download_root=DOWNLOAD_ROOT)
            model_path = os.path.join(DOWNLOAD_ROOT, ar + '.pt')
            clip_model = load_openai_model(model_path, device, jit=False)
            clip_model.float()
            clip_models.append(clip_model)
            # self.preprocess.append(preprocess)
            # self.resolutions.append(clip_model.visual.input_resolution)
            self.resolutions.append(clip_model.visual.image_size)
            weights.append(CONFIDECES[ar])

        self.clip_models = nn.ModuleList(clip_models)
        self.n_model = len(self.clip_models)
        self.weights = [round(x / sum(weights), 2) for x in weights]

        self.clipscore_weight = clipscore_weight
        self.device = device
        self.classification = classification
        self.class_features = None
        self.image_features = None
        self.amplify_rewards = amplify_rewards
        self.sample_k = sample_k
        self.reward_process = reward_process
        self.process_batch = process_batch
        self.weighted_scores = weighted_scores

        self.clip_models.eval()

        print("\n CLIPRewardsMultiple model created: \n"
                "\t visual backbone: {}, resolutions: {}, weighted_scores / weights: [ {} / {} ] \n"
                "\t amplify_rewards: {}, sample_k: {}, reward_process: {}, process_batch: {}\n".format(
                    arch, self.resolutions, weighted_scores, self.weights,
                    amplify_rewards, sample_k, reward_process, process_batch))

    @torch.no_grad()
    def CLIPScore(self, text_index=None, images_index=None, pairwise=True):
        """
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        all_scores = []
        for i in range(self.n_model):
            # suitable for image-2-text retrieval
            text_features = self.text_features[i][text_index] if text_index is not None else \
                                torch.repeat_interleave(self.text_features[i], self.sample_k, dim=0)
            # suitable for text-2-image retrieval
            image_features = self.image_features[i][images_index] if images_index is not None else \
                                torch.repeat_interleave(self.image_features[i], self.sample_k, dim=0)

            if pairwise:
                similarity = self.clipscore_weight * text_features @ image_features.t()
            else:
                similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

            # [n_samples]
            scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()
            all_scores.append(scores)

        scores = torch.stack(all_scores, dim=0)
        # [n_samples]
        if self.weighted_scores:
            weights = torch.tensor(self.weights, device=scores.device, dtype=scores.dtype).unsqueeze(1)
            final_scores = torch.sum(weights * scores, dim=0)
        else:
            final_scores = torch.mean(scores, dim=0)

        return final_scores

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
