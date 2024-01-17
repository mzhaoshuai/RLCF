# coding=utf-8
import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Test Time Adaptation for Retrieval Task")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp",
                            help="Floating point precition.")

    parser.add_argument('--output', type=str, default='tta_ret_rl_01', help='the output path')
    parser.add_argument('--retrieval_task', type=str, default="image2text", choices=["image2text", "text2image"],
                            help='using simple average or exponential average for gradient update')
    parser.add_argument('--arch', metavar='ARCH', default='ViT-B-16')

    # RL config
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--sample_k', type=int, default=5)
    parser.add_argument('--multiple_reward_models', type=int, default=0)
    parser.add_argument('--reward_arch', type=str, default='ViT-L-14')
    parser.add_argument('--reward_process', type=int, default=1,
                         help='If true, process rewards (raw CLIPScore)')
    parser.add_argument('--process_batch', type=int, default=0,
                         help='If true, process rewards through the whole batch (augmentations from a single images)')
    parser.add_argument('--reward_amplify', type=int, default=0)
    parser.add_argument('--weighted_scores', type=int, default=1)

    # args of momentum_update
    parser.add_argument('--momentum_update', type=int, default=0,
                         help='If true, update the model in a momentum fashion')
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_w', type=float, default=1.0)
    parser.add_argument('--tta_momentum', type=float, default=0.9999)
    
    # LAVIS confif
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    print('\n', vars(args), '\n')
    save_hp_to_json(args.output, args)

    return args


def save_hp_to_json(directory, args):
    """Save hyperparameters to a json file
    """
    filename = os.path.join(directory, 'hparams_{}.json'.format(args.retrieval_task))
    hparams = vars(args)
    with open(filename, 'w') as f:
        json.dump(hparams, f, indent=4, sort_keys=True)
