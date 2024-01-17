# coding=utf-8
import os
import json
import torch
import os.path
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from transformers import AutoTokenizer

from clip_reward import get_reward_model
from image_llm.clip import clip
from image_llm.params import get_args
from image_llm.custom_models import CAP_TTA
from image_llm.datasets import COCOCapDatasetForEmbedding, Flickr30kCapDatasetForEmbedding, NocapsCapDatasetForEmbedding
from image_llm.models.generate_opt import generate as opt_generate
from image_llm.models.modules import ClipCaptionPrefixV2, MappingType
from image_llm.utils import save_commandlines, get_device, save_checkpoint


class TxtLogger():
    def __init__(self, log_file) -> None:
        self.log_file = log_file
        with open(self.log_file , "w") as file:
            file.write(log_file + "\n\n")

    def log_id(self, image_id):
        with open(self.log_file , "a") as file:
            file.write(image_id + "\n")

    def log_sample_text(self, sample_text, scores):
        with open(self.log_file , "a") as file:
            for t in sample_text:
                file.write(t + "\n")
            for s in scores:
                file.write(str(round(s, 4)) + "  ")
            file.write("\n")

    def log_final_text(self, final_text):
        with open(self.log_file , "a") as file:
            file.write("final text:" + "\n")
            for t in final_text:
                file.write(t + "\t")
            file.write("\n")
            file.write("\n")


def make_preds_policy_batch(model, reward_model, tokenizer, device, optimizer=None, optim_state=None,
                            batch_size=16, scaler=None, args=None):
    text_logger = TxtLogger(args.out_results_file.replace(".json", ".txt"))
    model = model.to(device)
    # clip model for extracting features
    clip_model, preprocess = clip.load(args.clip_model_type, device=device, jit=False, download_root=args.download_root)
    clip_model.float()
    clip_model.eval()

    # dataset
    if args.dataset_mode == 0:
        dataset = COCOCapDatasetForEmbedding(args.annotations, args.images_root, preprocess, split='test')
    elif args.dataset_mode == 1:
        dataset = Flickr30kCapDatasetForEmbedding(args.annotations, args.images_root, preprocess, split='test')
    elif args.dataset_mode == 2:
        dataset = NocapsCapDatasetForEmbedding(args.annotations, args.images_root, preprocess, split='test')
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    sample_k = reward_model.sample_k
    results = []
    results_clipscore = dict()

    for i, sample in enumerate(tqdm(dataloader, ncols=150)):
        image, captions, image_ids, sub_paths = sample['image'].to(device), sample['caption'], sample['image_id'], sample['image_subpath']

        # extract CLIP image embedding
        with torch.no_grad():
            model.eval()
            with torch.cuda.amp.autocast():
                bs_prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                if args.normalize_prefix: bs_prefix = bs_prefix / bs_prefix.norm(2, -1, keepdim=True)

        for j in range(bs_prefix.shape[0]):
            reward_model.set_image_features(images=image[j:j+1, ...])
            image_id = image_ids[j].item()
            prefix = bs_prefix[j : j + 1]
            repeat_prefix = prefix.repeat(sample_k, 1)
            text_logger.log_id(sub_paths[j])

            # policy gradient for single image
            for step in range(args.tta_steps):
                # generate text
                with torch.no_grad():
                    model.eval()
                    with torch.cuda.amp.autocast():
                        prefix_embed = model.clip_project(prefix).reshape(prefix.shape[0], args.prefix_length, -1)
                        sampled_text = opt_generate(model.llm.llm, tokenizer, prompt="", query_embeds=prefix_embed,
                                                num_beams=sample_k, device=device, num_captions=sample_k,
                                                use_nucleus_sampling=args.use_nucleus_sampling > 0)
                        reward_model.set_text_features(captions=sampled_text)
                        # rewards calculation and process (note we process single image)
                        clip_score = reward_model.CLIPScore(text_index=torch.arange(sample_k, dtype=torch.long, device=device), pairwise=False)
                        rewards = reward_model.rewards_post_process(clip_score if reward_model.process_batch else clip_score.reshape(1, -1))
                        # log
                        text_logger.log_sample_text(sampled_text, rewards.tolist())

                model.train()
                optimizer.zero_grad()

                return_tokens = tokenizer(sampled_text, return_tensors="pt", padding=True).to(device)
                tokens = return_tokens.input_ids
                # generate attention mask
                atts_opt = torch.ones((len(sampled_text), args.prefix_length), dtype=torch.long).to(device)
                attention_mask = torch.cat([atts_opt, return_tokens.attention_mask], dim=1)
                # print(attention_mask.shape)

                with torch.cuda.amp.autocast():
                    outputs = model(tokens, repeat_prefix, attention_mask)
                    logits = outputs.logits[:, args.prefix_length - 1: -1]
                    # tokens as label
                    all_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(),
                                                ignore_index=0, reduction='none').reshape(logits.shape[0], -1)
                    loss = torch.mean(rewards * all_loss.mean(dim=-1))
                    # import pdb; pdb.set_trace()

                # compute gradient and do SGD step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # generate text
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    prefix_embed = model.clip_project(prefix).reshape(prefix.shape[0], args.prefix_length, -1)
                    generated_text_prefix = opt_generate(model.llm.llm, tokenizer, prompt="", query_embeds=prefix_embed,
                                                            num_beams=5, device=device, num_captions=1)
            # save resutls
            text_logger.log_final_text(generated_text_prefix)
            results.append({"image_id": image_id,
                            "caption": generated_text_prefix[0].lower()}
                            )
            results_clipscore[sub_paths[j]] = generated_text_prefix[0].lower()
            # reset/update parameters and optimizer state
            model.momentum_update_model()
            model.reset_initial()
            optimizer.load_state_dict(optim_state)            

    print("save results to {}".format(args.out_results_file))
    with open(args.out_results_file, 'w') as outfile:
        json.dump(results, outfile)

    print("save results to {}".format(args.out_clipscore_file))
    with open(args.out_clipscore_file, 'w') as outfile:
        json.dump(results_clipscore, outfile)


def main():
    args = get_args()
    save_commandlines(os.path.dirname(args.out_results_file), args, "train_policy_commandline_args.txt")
    device = get_device(0)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_config_dir, use_fast=False)

    mapping_type = {'mlp': MappingType.MLP,
                    'transformer': MappingType.Transformer,
                    'transformer_encoder': MappingType.TransformerEncoder,
                    'transformer_decoder': MappingType.TransformerDecoder
                    }[args.mapping_type]

    cap_model = ClipCaptionPrefixV2(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=args.prefix_dim,
                                num_layers=args.num_layers, mapping_type=mapping_type,
                                config_dir=args.llm_config_dir,
                                clip_patch=args.clip_patch
                                )

    # create model
    model = CAP_TTA(cap_model, device, momentum_update=args.momentum_update, update_freq=args.update_freq,
                            update_w=args.update_w, momentum=args.tta_momentum, cap_ckpt=args.checkpoint)
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # define the CLIPRewards
    reward_model = get_reward_model(device, args)

    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.tta_lr, eps=1e-06, weight_decay=args.tta_weight_decay)
    optim_state = deepcopy(optimizer.state_dict())

    make_preds_policy_batch(model, reward_model, tokenizer, device, optimizer=optimizer, optim_state=optim_state,
                                batch_size=16, scaler=scaler, args=args)

if __name__ == '__main__':
    main()