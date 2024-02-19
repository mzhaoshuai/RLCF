# coding=utf-8
# import os
# import json
# import logging
import argparse


def get_args(description='Caption'):
    """config of program"""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--data', default='clip_embedding.pkl',
                            help='path to clip embeddings of captions generated by the attached embeddings_generator script')
    parser.add_argument('--checkpoint', default=f'./checkpoints/coco_prefix_t10_rn-006.pt')
    # 0 for coco val, 1 for flicker30, 2 humor style,3 romantic,4 factual of style, 
    # 5 coco val text only, 6 coco train, 7 coco val for womanSnowboard_for_creating_capdec_preds
    parser.add_argument('--dataset_mode', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='COCO', choices=['COCO', 'CC12M'])
    parser.add_argument('--annotations', type=str, default=None,
                            help='the annotation file path of dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')
    # precision of training weights
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp",
                            help="Floating point precition.")

    parser.add_argument('--use_image_embedding', dest='use_image_embedding', action='store_true', default=False,
                            help='use image embedding as ClipCap')
    parser.add_argument('--images_root', type=str, default=None,
                            help='root for image locations')

    parser.add_argument('--clip_model_type', default='ViT-B/16', choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'))
    parser.add_argument('--download_root', type=str, default=None, help="clip download root")
    parser.add_argument('--clip_patch', type=int, default=0, help='whether use CLIP patch tokens')

    parser.add_argument('--cap_model', default="CapDec", choices=('CLIPCap', 'CapDec'))
    # parser.add_argument('--token_filename', default="gpt2", help="prefix name of the token file")

    parser.add_argument('--resume', default=None,
                            help='path to resume weights, if not specified, will train from scratch')
    parser.add_argument('--out_dir', default='./checkpoints', help='path to output directory')
    parser.add_argument('--out_results_file', type=str, default='the output file save the generation file')
    parser.add_argument('--out_clipscore_file', type=str, default='the output file save the generation file')

    parser.add_argument('--add_modality_offset', dest='add_modality_offset', action='store_true', default=False,
                            help='train with modality offset that was pre calculated at others/CLIP_embeddings_centers_info.pkl')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--noise_variance', type=float, default=0.0, help='noise variance')

    parser.add_argument('--uniform_noise', dest='uniform_noise', action='store_true', default=False,
                            help='use uniform noise instead of gaussian')

    parser.add_argument('--bs', type=int, default=34, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=5000,
                            help='warm up steps')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1, help='save every n epochs')

    parser.add_argument('--prefix_length', type=int, default=40, help='prefix length')
    parser.add_argument('--prefix_length_clip', type=int, default=40,
                            help='prefix length for clip')

    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true', default=False,
                            help='train only the mapper between CLIP and LLM, while LLM is frozen')
    parser.add_argument('--mapping_type', type=str, default='transformer',
                            help='type of architurctre between CLIP and LLM (mlp/transformer)')
    parser.add_argument('--num_layers', type=int, default=8, help='number of layers in the mapper')
    parser.add_argument('--llm_config_dir', type=str, default=None,
                            help='config and pretrained files of GPT2')

    parser.add_argument('--beam', dest='beam', action='store_true', default=True,
                            help='whether use beam search')

    parser.add_argument('--use_nucleus_sampling', type=int, default=0)
    # configs for RL
    parser.add_argument('--tta_steps', type=int, default=5, help='number of policy training steps')
    parser.add_argument('--tta_lr', type=float, default=1e-5, help='learning rate of policy gradient')
    parser.add_argument('--tta_weight_decay', default=5e-4, type=float)

    parser.add_argument('--sample_k', type=int, default=5)
    parser.add_argument('--multiple_reward_models', type=int, default=0)
    parser.add_argument('--reward_arch', type=str, default='ViT-L/14')
    parser.add_argument('--reward_process', type=int, default=1,
                         help='If true, process rewards (raw CLIPScore)')
    parser.add_argument('--process_batch', type=int, default=0,
                         help='If true, process rewards through the whole batch (augmentations from a single images)')
    parser.add_argument('--reward_amplify', type=int, default=0)
    parser.add_argument('--weighted_scores', type=int, default=1)

    parser.add_argument('--momentum_update', type=int, default=0,
                         help='If true, update the model in a momentum fashion')
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_w', type=float, default=1.0)
    parser.add_argument('--tta_momentum', type=float, default=0.9999)

    args = parser.parse_args()

    if 'RN' in args.clip_model_type:
        args.prefix_dim = 640
    elif 'ViT-L' in args.clip_model_type:
        args.prefix_dim = 768
    elif 'ViT-B' in args.clip_model_type:
        args.prefix_dim = 512
    else:
        raise NotImplementedError

    print('\n', vars(args), '\n')

    return args