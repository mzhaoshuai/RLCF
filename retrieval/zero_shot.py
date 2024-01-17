# coding=utf-8
import os
import json
import logging
import torch

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from params import parse_args
from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_Multiple


def test_time_tune(data_loader, model, text_bs=128):
    """policy gradient when playing image to text policy"""
    model.eval()
    device = model.device

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model.set_many_text_features(data_loader.dataset.text, text_bs=text_bs)
            model.set_image_features_with_dataloder(data_loader)
            score_matrix_i2t, score_matrix_t2i = model()

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
    model = CLIPRet_Multiple(device)
    model = model.to(device)

    score_matrix_i2t, score_matrix_t2i = test_time_tune(dataloader, model, text_bs=128)
    eval_result = task._report_metrics(score_matrix_i2t, score_matrix_t2i, dataloader.dataset.txt2img, dataloader.dataset.img2txt)

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
