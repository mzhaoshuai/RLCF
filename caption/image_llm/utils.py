# coding=utf-8
import os
import math
import json
import torch
import shutil
import argparse
import numpy as np


def get_uniform_ball_noise(input_shape, radius=0.1, device='cpu'):
    # normal distribution
    uniform_noise_ball = torch.randn(input_shape, device=device)
    uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=1)

    # unified distribution
    u = torch.rand(input_shape[0], device=device)  
    u = u ** (1. / input_shape[1])
    uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T

    return uniform_noise_ball


def noise_injection(x, variance=0.001, modality_offset=None, uniform_noise=False, dont_norm=False, device='cpu'):
    if variance <= 0.0: return x
    std = math.sqrt(variance)

    if not dont_norm:
        x = torch.nn.functional.normalize(x, dim=1)

    if uniform_noise:
        x = x + get_uniform_ball_noise(x.shape, radius=std)
    else:
        # todo by some conventions multivraiance noise should be devided by sqrt of dim
        x = x + (torch.randn(x.shape, device=device) * std)

    # add pre-calculated modality gap
    # if modality_offset is not None:
    #     x = x + modality_offset
    
    return torch.nn.functional.normalize(x, dim=1)


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")

    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


class Timer:
    """
    measure inference time
    """
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.timings = []

    def __enter__(self):
        self.starter.record()
        return self

    def __exit__(self, *args):
        self.ender.record()
        torch.cuda.synchronize()
        interval = self.starter.elapsed_time(self.ender)
        self.timings.append(interval)
        self.sum += interval
        self.count += 1

    def __str__(self):
        mean_syn = self.sum / self.count
        std_syn = np.std(self.timings)
        return f"mean: {mean_syn:.2f} ms, std: {std_syn:.2f} ms"


def save_commandlines(out_dir, args, filename="commandline_args.txt"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(f'{out_dir}/{filename}', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(f'args saved to file {out_dir}/{filename}')


def save_checkpoint(state, is_best, model_dir, filename='checkpoint.pth.tar'):
	filename = os.path.join(model_dir, filename)
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def get_device(device_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')

    device_id = min(torch.cuda.device_count() - 1, device_id)
    
    return torch.device(f'cuda:{device_id}')
