# coding=utf-8
import os
import clip
import torch
import torch.nn as nn

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
        "ViT-L/14@336px": 10,
        "ViT-L/14": 5,
        "RN50x64": 3,
        "ViT-B/16": 1
    }


def get_reward_model(device, args):
    if args.multiple_reward_models:
        reward_model = CLIPRewardsMultiple(device, arch=["ViT-L/14@336px", "RN50x64", "ViT-L/14"], classification=True,
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

    @torch.no_grad()
    def extract_image_features(self, images):
        pass

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        pass

    @torch.no_grad()
    def set_class_features(self, classnames=None, tokenized_classes=None):
        self.class_features = self.extract_text_features(captions=classnames, tokenized_cap=tokenized_classes)

    @torch.no_grad()
    def set_image_features(self, images):
        self.image_features = self.extract_image_features(images)

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


class CLIPRewards(BaseRewards):
    def __init__(self, device, arch="ViT-B/16", clipscore_weight=2.5, classification=True,
                    amplify_rewards=False, sample_k=5, reward_process=True, process_batch=False,
                    default_resolutions=224) -> None:
        """
        calculating CLIP Reward
        Args:
            clipscore_weight: weight for calculating CLIPScore
            reward_process: If ture, post-process the rewards, e.g., subtract the reward mean or standardization
            amplify_rewards: If true, after subtracting the reward mean, also divide rewards by standard variance of rewards, i.e, standardization.
            sample_k: K for sampling.
            process_batch: If true, post-process the rewards within the {BatchSize x K} sampled text-image pairs.
                Others, post-process the rewards within the {1 x K} sampled text-image pairs.
                TPT augment the images, so we have a batch of augmented images from a single image.
        """
        super().__init__()
        self.default_resolutions = default_resolutions
        self.clip_model, self.embed_dim, self.preprocess = clip.load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.resolutions = self.clip_model.visual.input_resolution
        self.clipscore_weight = clipscore_weight
        self.device = device
        self.classification = classification
        self.class_features = None
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
    def CLIPScore(self, class_index, images=None, image_features=None, captions=None, tokenized_cap=None, text_features=None,
                        pairwise=True):
        """
        class_index: sampled class index
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        text_features = self.class_features[class_index]
        image_features = torch.repeat_interleave(self.image_features, self.sample_k, dim=0)

        if pairwise:
            similarity = self.clipscore_weight * text_features @ image_features.t()
        else:
            similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

        scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()

        return scores

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features without normalization"""
        if self.resolutions != self.default_resolutions:
            images = nn.functional.interpolate(images, size=self.resolutions, mode='bicubic', align_corners=True)
        image_features = self.clip_model.encode_image(images).float()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        if captions is not None:
            caption_tokens = clip.tokenize(captions, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(caption_tokens).float()
        if tokenized_cap is not None:
            text_features = self.clip_model.encode_text(tokenized_cap).float()

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    @torch.no_grad()
    def rewards_post_process(self, clip_score):
        """
        clip_score: shape [bs, K] or [bs * K]
        """
        if clip_score.shape[-1] > 1 and self.reward_process:
            mean = torch.mean(clip_score, dim=-1, keepdim=True)
            if self.amplify_rewards:
                std = torch.std(clip_score, dim=-1, keepdim=True) + 1e-5
            else:
                std = 1.0
            clip_score = (clip_score - mean) / std

        return clip_score.flatten()

    @torch.no_grad()
    def calulate_similarity(self):
        """
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * self.image_features @ self.class_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class CLIPRewardsMultiple(BaseRewards):
    def __init__(self, device, arch=["ViT-B/16", "RN50x64", "ViT-L/14"], clipscore_weight=2.5, classification=True,
                    amplify_rewards=False, sample_k=5, reward_process=True, process_batch=True, weighted_scores=True,
                    default_resolutions=224) -> None:
        """
        calculating CLIP Reward using multiple CLIP models
        Args:
            arch: a list of CLIP arches
            clipscore_weight: weight for calculating CLIPScore
            weighted_scores: if true, the final score is the weighted average of all scores
        """
        super().__init__()
        # load clip models
        clip_models = []
        self.preprocess = []
        self.resolutions = []
        weights = []
        self.default_resolutions = default_resolutions
        for ar in arch:
            clip_model, embed_dim, preprocess = clip.load(ar, device=device, download_root=DOWNLOAD_ROOT)
            clip_models.append(clip_model)
            self.preprocess.append(preprocess)
            self.resolutions.append(clip_model.visual.input_resolution)
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
    def CLIPScore(self, class_index, images=None, image_features=None, captions=None, tokenized_cap=None, text_features=None,
                        pairwise=True):
        """
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        all_scores = []
        for i in range(self.n_model):
            text_features = self.class_features[i][class_index]
            image_features = torch.repeat_interleave(self.image_features[i], self.sample_k, dim=0)

            if pairwise:
                similarity = self.clipscore_weight * text_features @ image_features.t()
                raise NotImplementedError
            else:
                similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

            # [n_samples]
            scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()
            all_scores.append(scores)

        scores = torch.stack(all_scores, dim=0)
        # [n_samples]
        if self.weighted_scores:
            # final_scores = torch.sum(scores / (torch.sum(scores, dim=0, keepdim=True) + 1e-5) * scores, dim=0)
            weights = torch.tensor(self.weights, device=scores.device, dtype=scores.dtype).unsqueeze(1)
            final_scores = torch.sum(weights * scores, dim=0)
        else:
            final_scores = torch.mean(scores, dim=0)

        return final_scores

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features with normalization"""
        image_features = []
        for i in range(self.n_model):
            if self.resolutions[i] != self.default_resolutions:
                # different CLIP has different input sizes
                tmp_images = nn.functional.interpolate(images, size=self.resolutions[i], mode='bicubic', align_corners=True)
            else:
                tmp_images = images
            image_feat = self.clip_models[i].encode_image(tmp_images).float()
            image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
            image_features.append(image_feat)

        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        """extract text features with normalization"""
        text_features = []
        for i in range(self.n_model):
            if captions is not None:
                caption_tokens = clip.tokenize(captions, truncate=True).to(self.device)
                text_feat = self.clip_models[i].encode_text(caption_tokens).float()

            if tokenized_cap is not None:
                text_feat = self.clip_models[i].encode_text(tokenized_cap).float()

            # normalized features
            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            text_features.append(text_feat)

        return text_features

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


if __name__ == "__main__":
    # The Figure 1 (b) in the paper
    import torchvision.transforms as transforms
    from PIL import Image
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    path = "/home/shuzhao/Data/dataset/test_images"
    device = torch.device('cuda:{}'.format(0))
    images = os.listdir(path)
    print(images)
    resolution = 224
    arch = "ViT-B/16"
    arch = "ViT-L/14"
    clipscore_weight = 2.5
    captions = [
                "There are three sheeps standing together on the grass.",
                "A group of baseball players is crowded at the mound.",
                "Two girls bathe an elephant lying on its side"
                ]

    clip_model, embed_dim, preprocess = clip.load(arch, device=device, download_root=DOWNLOAD_ROOT)
    clip_model = clip_model.float()
    clip_model.eval()

    all_iamges = []
    for file in images:
        image = Image.open(os.path.join(path, file))
        all_iamges.append(preprocess(image))
    images = torch.stack(all_iamges, dim=0).to(device)

    with torch.no_grad():
        # images
        image_features = clip_model.encode_image(images).float()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # captions
        caption_tokens = clip.tokenize(captions, truncate=True).to(device)
        text_features = clip_model.encode_text(caption_tokens).float()
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        similarity = clipscore_weight * text_features @ image_features.t()

    print(similarity)

    mean = torch.mean(similarity, dim=0, keepdim=True)
    print(similarity - mean)
    # ['COCO_val2014_000000001164.jpg', 'COCO_val2014_000000000772.jpg', 'COCO_val2014_000000000192.jpg']
    # tensor([[0.4146, 0.7624, 0.4753],
    #         [0.3114, 0.4829, 0.6724],
    #         [0.8394, 0.3277, 0.2738]], device='cuda:0')

    # CLIP-ViT-L/14
    # ['COCO_val2014_000000001164.jpg', 'COCO_val2014_000000000772.jpg', 'COCO_val2014_000000000192.jpg']
    # tensor([[0.0721, 0.6127, 0.2376],
    #         [0.0638, 0.2741, 0.3465],
    #         [0.7014, 0.2067, 0.0213]], device='cuda:0')
    # tensor([[-0.2070,  0.2482,  0.0358],
    #         [-0.2153, -0.0904,  0.1447],
    #         [ 0.4223, -0.1578, -0.1805]], device='cuda:0')
