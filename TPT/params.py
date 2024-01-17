# coding=utf-8
import os
import json
import argparse


def none_or_str(value):
    if value == 'None':
        return None
    return value


def get_args(description='Test Time Reinforcement Learning With CLIP Reward'):
    """config of program"""
    parser = argparse.ArgumentParser(description=description)

    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('--output', type=str, default='exp_01', help='the output path')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('-p', '--print-freq', default=500, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=none_or_str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--hard_aug', type=int, default=0,
                            help='If true, use a hard augmentation')

    parser.add_argument('--augmix', type=int, default=1,
                            help='If true, use augmix augmentation')

    # RL config
    parser.add_argument('--sample_k', type=int, default=5)
    parser.add_argument('--multiple_reward_models', type=int, default=0)
    parser.add_argument('--reward_arch', type=str, default='ViT-L/14')
    parser.add_argument('--reward_process', type=int, default=1,
                         help='If true, process rewards (raw CLIPScore), for example, baseline subtraction')
    parser.add_argument('--process_batch', type=int, default=0,
                         help='If true, process rewards through the whole batch (augmentations from a single images)')
    parser.add_argument('--reward_amplify', type=int, default=0)
    parser.add_argument('--weighted_scores', type=int, default=1)

    # --confidence_gap and --min_entropy_reg are only experimental features
    parser.add_argument('--confidence_gap', type=int, default=0)
    parser.add_argument('--confidence_gap_w', type=float, default=0.5)
    parser.add_argument('--min_entropy_reg', type=int, default=0)
    parser.add_argument('--min_entropy_w', type=float, default=0.1)

    parser.add_argument('--momentum_update', type=int, default=0,
                            help='If true, update the model in a momentum fashion')
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_w', type=float, default=1.0)
    parser.add_argument('--tta_momentum', type=float, default=0.9999)
    parser.add_argument('--tune_norm', type=int, default=0)

    # for BN adaptation in CLIP ResNet series model
    parser.add_argument('--prior_strength', type=int, default=-1,
                            help="Adapting BN statistics, the strength will used to calculate the momentum coefficient")

    # For ImageNet-C
    parser.add_argument('--corruption', type=str, default='defocus_blur', help="corruption type")
    parser.add_argument('--level', type=str, default='5', help="Corruption Level")

    # For KD loss choice
    parser.add_argument('--kd_loss', type=str, default='KD', choices=["KD", "DKD", "ATKD"],
                            help="choices of KD loss types")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None

    print('\n', vars(args), '\n')
    save_hp_to_json(args.output, args)

    return args


def save_hp_to_json(directory, args):
    """Save hyperparameters to a json file
    """
    filename = os.path.join(directory, 'hparams_train.json')
    hparams = vars(args)
    with open(filename, 'w') as f:
        json.dump(hparams, f, indent=4, sort_keys=True)
