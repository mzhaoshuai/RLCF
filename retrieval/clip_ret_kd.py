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
from retrieval.clip_reward import get_reward_model
from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_TTA

from clip_ret_policy import tokenize_all_text, get_all_text_embeds, get_all_image_embeds


def kd_distill_loss_v2(logits_student, logits_teacher, T_stu=1.0, T_tea=1.0):
    """
    vanilla KD, KLDiv between teacher and student, only the gradient related part
    """
    log_pred_student = F.log_softmax(logits_student / T_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_tea, dim=1)
    # kl_div = -p log q
    loss_kd = - torch.sum(pred_teacher * log_pred_student, dim=1).mean()
    loss_kd = loss_kd * T_stu * T_stu

    return loss_kd


def tune_image(image, model, reward_model, optimizer, scaler, args=None):
    """tune function for test time adaptation (image encoder)"""
    model.train()

    # knowledge distillation for single sample
    reward_model.set_image_features(images=image)
    r_logits_per_image, r_logits_per_text = reward_model.calulate_similarity()

    for step in range(args.tta_steps):    
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text = model(image)
            loss = kd_distill_loss_v2(logits_per_image, r_logits_per_image)

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
    model.train()

    # knowledge distillation for single sample
    reward_model.set_text_features(captions=text)
    r_logits_per_image, r_logits_per_text = reward_model.calulate_similarity()

    for step in range(args.tta_steps):   
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text = model(images=None, text=text)
            loss = kd_distill_loss_v2(logits_per_text, r_logits_per_text)

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
