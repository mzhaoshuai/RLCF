# coding=utf-8
import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


from image_llm.params import get_args
from image_llm.datasets import get_dataset
from image_llm.utils import noise_injection, save_commandlines, get_device, save_checkpoint
from image_llm.models.modules import MappingType, ClipCaptionPrefixV2


def train(dataset, model, args, device=torch.device('cuda:0'),
            scaler=None, optimizer=None, train_dataloader=None, scheduler=None, start_epoch=0):
    model = model.to(device)
    model.train()
    modality_offset = None

    loss_per_epoch_train = []
    loss_per_epoch_val = []
    for epoch in range(start_epoch, args.epochs):
        print(f">>> Training epoch {epoch} / {args.epochs}")
        progress = tqdm(total=len(train_dataloader), ncols=100)
        accumulated_loss = 0.0

        for idx, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            scheduler.step()

            tokens, mask, prefix = sample['tokens'].to(device), sample['mask'].to(device), sample['prefix'].to(device, dtype=torch.float32)
            with torch.cuda.amp.autocast():
                # add noise to prefix, in CapDec, prefix is the text embedding from CLIP
                if args.cap_model == "CapDec":
                    prefix = noise_injection(prefix, args.noise_variance,
                                                modality_offset=modality_offset, uniform_noise=args.uniform_noise,
                                                dont_norm=args.normalize_prefix,
                                                device=device)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                # tokens as label
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

            # update weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress.set_postfix({"loss": loss.item()})
            progress.update()
            accumulated_loss += loss.item()
        progress.close()

        loss_per_epoch_train.append(accumulated_loss / len(train_dataloader))
        print('loss_per_epoch_train: ', loss_per_epoch_train)

        # save checkpoint
        ckpt_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
        if scaler is not None: ckpt_dict['scaler'] = scaler.state_dict()
        save_checkpoint(ckpt_dict, False, args.out_dir, filename=f"ckpt-latest.pt")
        if epoch in range(args.epochs - 6, args.epochs):
            save_checkpoint(ckpt_dict, False, args.out_dir, filename=f"ckpt-{epoch:03d}.pt")

    with open(os.path.join(args.out_dir, f"loss_per_epoch.json"), 'w') as f:
        json.dump({'train': loss_per_epoch_train, 'val': loss_per_epoch_val}, f)

    return model


def main():
    args = get_args()
    save_commandlines(args.out_dir, args, "train_commandline_args.txt")
    device = get_device(0)

    # create dataset
    dataset = get_dataset(args)

    # load model
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]

    # we always fix the LLM
    model = ClipCaptionPrefixV2(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=args.prefix_dim,
                                    num_layers=args.num_layers, mapping_type=args.mapping_type,
                                    config_dir=args.llm_config_dir, clip_patch=args.clip_patch)

    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-06, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    train_dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.num_workers)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                    num_training_steps=args.epochs * len(train_dataloader))

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            sd = checkpoint["state_dict"]
            # if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            # 	sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            start_epoch = checkpoint['epoch']
            if "optimizer" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "scaler" in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])

    train(dataset, model, args, device=device, scaler=scaler, optimizer=optimizer,
            train_dataloader=train_dataloader, scheduler=scheduler, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
