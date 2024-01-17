# coding=utf-8
import os
import json
import time
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.cls_to_names import *
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.datautils import AugMixAugmenter, build_dataset
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from params import get_args
from clip.custom_clip import CLIPCLS_TTA_Multiple, CLIPCLS_TTA


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)

    print("Use GPU: {} for training".format(args.gpu))
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        device = torch.device('cpu')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes

    # create model
    # model = CLIPCLS_TTA_Multiple(device, classnames, prompt_prefix=args.ctx_init, default_resolutions=args.resolution)
    model = CLIPCLS_TTA(device, classnames, arch=args.arch, prompt_prefix=args.ctx_init)
    model = model.cuda(args.gpu)

    # define optimizer
    trainable_param = model.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr, weight_decay=args.weight_decay)
    optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = None
    print('=> Using native Torch AMP. Training in mixed precision.')

    torch.backends.cudnn.benchmark = True
    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    all_start = time.time()
    for set_id in datasets:
        data_start_time = time.time()

        data_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
        batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # reset the model, Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I', 'C']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        model.reset_classnames_and_state(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode,
                                        corruption=args.corruption, level=args.level)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                                                    num_workers=args.workers, pin_memory=True)
 
        results[set_id] = test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args, device=device)
        del val_dataset, val_loader
        print("=> Acc. on testset [{}]: @1 {} / @5 {}".format(set_id, results[set_id][0], results[set_id][1]))      

        data_time = time.time() - data_start_time
        print('The running time for dataset {} is {:.1f} Hour {:.1f} Minute\n'.format(set_id, data_time // 3600, data_time % 3600 / 60))

    # save output
    with open(os.path.join(args.output, "results.json"), 'w') as fp:
        json.dump(results, fp, indent=4)
    print("======== Result Summary ========")
    print(args.output)
    print(results)

    all_time = time.time() - all_start
    print('The total running time of the program is {:.1f} Hour {:.1f} Minute\n'.format(all_time // 3600, all_time % 3600 / 60))
    print('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
                torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))


def test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args, device=None, reward_model=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix='Test: ')
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                    output = model(image)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [round(x, 3) for x in [top1.avg.item(), top5.avg.item()]]


if __name__ == '__main__':
    args = get_args()

    set_random_seed(args.seed)
    main_worker(args.gpu, args)
    os.system('export ghost="cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook"')
