# coding=utf-8
import os
import json
import time
import random
import logging
import datetime
from tqdm import tqdm
from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now
from lavis.common.logger import MetricLogger

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from params import parse_args
from clip_reward import get_reward_model
from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_TTA
from lavis.models.clip_models.tokenizer import tokenize


def tokenize_all_text(texts, model, text_bs=128):
    """tokenize all text and return: (text_ids)"""
    num_text = len(texts)
    text_ids = []
    i = 0
    while i < num_text:
        text = texts[i : min(num_text, i + text_bs)]
        input_ids = tokenize(text).to(model.device)
        text_ids.append(input_ids)
        i += text_bs
    text_ids = torch.cat(text_ids, dim=0)

    return text_ids


def get_all_text_embeds(text_inputs, model, text_bs=128):
    logging.info("Extracting ALL Text features...")
    text_embeds = []
    i = 0
    while i < text_inputs.shape[0]:
        batch = text_inputs[i : min(text_inputs.shape[0], i + text_bs)]
        text_features = model.get_text_features(text=None, tokenized_prompts=batch)
        text_embeds.append(text_features)
        i += text_bs

    return torch.cat(text_embeds, dim=0)


def get_all_image_embeds(data_loader, model):
    """extract all image embeddings"""
    logging.info("Extracting ALL image features...")
    image_embeds = []
    for samples in data_loader:
        image = samples["image"].to(model.device)
        image_features = model.get_image_features(image)
        image_embeds.append(image_features)

    return torch.cat(image_embeds, dim=0)


def tune_image(image, model, reward_model, optimizer, scaler, args=None):
    """tune function for test time adaptation (image encoder)"""
    sample_k = reward_model.sample_k
    bs = image.shape[0]
    model.train()

    # policy gradient for single sample
    reward_model.set_image_features(images=image)
    for step in range(args.tta_steps):    
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text = model(image)
            # sample results
            value, index = torch.topk(logits_per_image, sample_k, dim=-1)
            text_index = index.flatten()
            clip_score = reward_model.CLIPScore(text_index=text_index, pairwise=False)
            rewards = reward_model.rewards_post_process(clip_score if reward_model.process_batch else clip_score.reshape(bs, -1))
            rep_output = torch.repeat_interleave(logits_per_image, sample_k, dim=0)

            # import pdb; pdb.set_trace()
            all_loss = F.cross_entropy(rep_output, text_index, reduction='none')
            loss = torch.mean(rewards * all_loss)

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def tune_text(text, model, reward_model, optimizer, scaler, args=None):
    """
    tune function for test time adaptation (text encoder)
    Args:
        text: raw text
    """
    sample_k = reward_model.sample_k
    bs = 1
    model.train()

    # policy gradient for single sample
    reward_model.set_text_features(captions=text)
    for step in range(args.tta_steps):    
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text = model(images=None, text=text)
            # sample results
            value, index = torch.topk(logits_per_text, sample_k, dim=-1)
            images_index = index.flatten()
            clip_score = reward_model.CLIPScore(text_index=None, images_index=images_index, pairwise=False)
            rewards = reward_model.rewards_post_process(clip_score if reward_model.process_batch else clip_score.reshape(bs, -1))
            rep_output = torch.repeat_interleave(logits_per_text, sample_k, dim=0)

            # import pdb; pdb.set_trace()
            all_loss = F.cross_entropy(rep_output, images_index, reduction='none')
            loss = torch.mean(rewards * all_loss)

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def test_time_tune(data_loader, device, model, reward_model=None, scaler=None, optimizer=None, optim_state=None, text_bs=128, args=None):
    """policy gradient when playing image to text policy"""
    model.eval()
    device = model.device

    # results matrix
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(data_loader.dataset.text)), -100.0).to(device)
    score_matrix_t2i = torch.full((len(data_loader.dataset.text), len(data_loader.dataset.image)), -100.0).to(device)
    image_filenames = [os.path.join(data_loader.dataset.vis_root, filename) for filename in data_loader.dataset.image]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            text_ids = tokenize_all_text(data_loader.dataset.text, model, text_bs)
            if model.only_visual:
                text_embeds = get_all_text_embeds(text_ids, model, text_bs)
                model.set_text_features(text_features=text_embeds)
                reward_model.set_many_text_features(data_loader.dataset.text, text_bs=text_bs)
            else:
                image_embeds = get_all_image_embeds(data_loader, model)
                model.set_image_features(image_features=image_embeds)
                reward_model.set_image_features_with_dataloder(data_loader)

    if model.only_visual:
        for i, samples in tqdm(enumerate(data_loader), ncols=150):
            image = samples["image"].to(model.device)
            tune_image(image, model, reward_model, optimizer, scaler, args=args)

            # after TTA, do evaluation
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # logits_per_image contains multiplication of logit_scale, it should not affect the retrieva results
                    logits_per_image, _ = model(image)
                score_matrix_i2t[i] = logits_per_image[0]

            # reset/update parameters
            model.momentum_update_model()
            model.reset_initial()
            # reset the optimizer state
            optimizer.load_state_dict(optim_state)
    else:
        for i, text in tqdm(enumerate(data_loader.dataset.text), ncols=150):
            tune_text(text, model, reward_model, optimizer, scaler, args=args)

            # after TTA, do evaluation
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _, logits_per_text = model(images=None, text=text)
                score_matrix_t2i[i] = logits_per_text[0]

            model.momentum_update_model()
            model.reset_initial()
            optimizer.load_state_dict(optim_state)

    return score_matrix_i2t.detach().cpu().numpy(), score_matrix_t2i.detach().cpu().numpy()


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    print('\n job_ID {}: \n'.format(job_id))

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    dataloader = runner.dataloaders['test']
    device = runner.model.device

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # create model
    model = CLIPRet_TTA(device, arch=args.arch, only_visual=(args.retrieval_task == "image2text"),
                            momentum_update=args.momentum_update, update_freq=args.update_freq,
                            update_w=args.update_w, momentum=args.tta_momentum)
    model = model.to(device)

    # define the CLIPRewards
    reward_model = get_reward_model(device, args)

    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-06, weight_decay=args.weight_decay)
    optim_state = deepcopy(optimizer.state_dict())

    # score_matrix_i2t, score_matrix_t2i = image_to_text_policy(dataloader, runner.model, optimizer, task.cfg.k_test, 128, args=args)
    score_matrix_i2t, score_matrix_t2i = test_time_tune(dataloader, device, model, reward_model=reward_model,
                                                scaler=scaler, optimizer=optimizer, optim_state=optim_state, text_bs=128, args=args)

    eval_result = task._report_metrics(score_matrix_i2t, score_matrix_t2i,
                                        dataloader.dataset.txt2img, dataloader.dataset.img2txt)

    # print and output
    output_filename = os.path.join(args.output, "results_{}.json".format(args.retrieval_task))
    logging.info(output_filename)
    for k, v in eval_result.items():
        eval_result[k] = round(v, 3)
    logging.info(eval_result)
    # save output
    with open(output_filename, 'w') as fp:
        json.dump(eval_result, fp, indent=4)


if __name__ == "__main__":
    main()
