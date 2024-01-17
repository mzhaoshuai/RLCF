import os
import numpy as np
import concurrent.futures
from PIL import Image
from typing import Tuple

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.hoi_dataset import BongardDataset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations


ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    'C': 'imagenet-c'
}


def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False,
                    corruption="defocus_blur", level="5"):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)

    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)

    elif set_id == 'C':
        print("dataset: ", ID_to_DIRNAME[set_id], corruption, level)
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], corruption, level)
        testset = datasets.ImageFolder(testdir, transform=transform)        
    
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment(hard_aug=False, resolution=224, crop_min=0.2):
    if hard_aug:
        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(crop_min, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomHorizontalFlip(),
            ])
    else:
        return transforms.Compose([
                transforms.RandomResizedCrop(resolution),
                transforms.RandomHorizontalFlip(),
            ])


def augmix(image, preaugment, preprocess, aug_list, severity=1):
    # preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, hard_aug=False):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = augmentations.augmentations if augmix else []
        self.severity = severity
        self.preaugment = get_preaugment(hard_aug=hard_aug, resolution=224, crop_min=0.2)
        print("\n AugMixAugmenter created: \n"
                "\t len(aug_list): {}, augmix: {} \n".format(len(self.aug_list), augmix))

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preaugment, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views


class AugMixAugmenterAsync(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, max_workers=8):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        # self.pool = mp.Pool(processes=self.max_workers)

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        futures = [self.executor.submit(augmix, x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        concurrent.futures.wait(futures)
        views = [future.result() for future in futures]

        return [image] + views

    def __del__(self):
        self.executor.shutdown()


if __name__ == "__main__":
    import time

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        # transforms.ToTensor(),
        normalize])

    data_transform1 = AugMixAugmenter(base_transform, preprocess, n_views=64-1, 
                                        augmix=False)

    data_transform2 = AugMixAugmenterAsync(base_transform, preprocess, n_views=64-1, 
                                            augmix=False)

    image = torch.rand(3, 512, 512)
    start = time.time()
    for i in range(10):
        data_transform1(image)
    end = time.time()
    print("The processing time is {} second".format(end - start))

    start = time.time()
    for i in range(10):
        res = data_transform2(image)
        # import pdb; pdb.set_trace()
    end = time.time()
    print("The processing time is {} second".format(end - start))
