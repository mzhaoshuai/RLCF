# coding=utf-8
import os
import json
import torch
import os.path
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer


from image_llm.params import get_args
from image_llm.clip import clip
from image_llm.generate import generate2, generate_beam
from image_llm.datasets import COCOCapDatasetForEmbedding, Flickr30kCapDatasetForEmbedding, NocapsCapDatasetForEmbedding
from image_llm.utils import save_commandlines, get_device
from image_llm.models.generate_opt import generate as opt_generate
from image_llm.models.modules import ClipCaptionPrefixV2, MappingType


@torch.no_grad()
def make_preds_batch(annotations, model, out_path, tokenizer, batch_size=10, args=None):
    device = get_device(0)
    model = model.to(device)
    model.eval()

    # clip model
    clip_model, preprocess = clip.load(args.clip_model_type, device=device, jit=False, download_root=args.download_root)
    clip_model.float()
    clip_model.eval()

    # dataset
    if args.dataset_mode == 0:
        dataset = COCOCapDatasetForEmbedding(annotations, args.images_root, preprocess, split='test')
    elif args.dataset_mode == 1:
        dataset = Flickr30kCapDatasetForEmbedding(annotations, args.images_root, preprocess, split='test')
    elif args.dataset_mode == 2:
        dataset = NocapsCapDatasetForEmbedding(annotations, args.images_root, preprocess, split='test')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    results = []
    results_clipscore = dict()

    for i, sample in enumerate(tqdm(dataloader, ncols=150)):
        image, captions, image_ids, sub_paths = sample['image'].to(device), sample['caption'], sample['image_id'], sample['image_subpath']

        with torch.cuda.amp.autocast():
            prefix = clip_model.encode_image(image, cls=not args.clip_patch).to(device, dtype=torch.float32)

            if args.normalize_prefix: prefix = prefix / prefix.norm(2, -1, keepdim=True)

            prefix_embed = model.clip_project(prefix).reshape(prefix.shape[0], args.prefix_length, -1)

            # generated_text_prefix = generate_beam(model, tokenizer, beam_size=5, embed=prefix_embed)[0]
            generated_text_prefix = opt_generate(model.llm.llm, tokenizer, prompt="",
                                                    query_embeds=prefix_embed, num_beams=5, device=device, num_captions=1)

        for j, idx in enumerate(image_ids):
            results.append({"image_id": idx.item(),
                            "caption": generated_text_prefix[j].lower()}
                            )
            results_clipscore[sub_paths[j]] = generated_text_prefix[j].lower()

    print("save results to {}".format(out_path))
    with open(out_path, 'w') as outfile:
        json.dump(results, outfile)

    print("save results to {}".format(args.out_clipscore_file))
    with open(args.out_clipscore_file, 'w') as outfile:
        json.dump(results_clipscore, outfile)


def main():
    args = get_args()
    save_commandlines(os.path.dirname(args.out_results_file), args, "pred_commandline_args.txt")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_config_dir, use_fast=False)

    mapping_type = {'mlp': MappingType.MLP,
                    'transformer': MappingType.Transformer,
                    'transformer_encoder': MappingType.TransformerEncoder,
                    'transformer_decoder': MappingType.TransformerDecoder
                    }[args.mapping_type]

    model = ClipCaptionPrefixV2(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=args.prefix_dim,
                                num_layers=args.num_layers, mapping_type=mapping_type,
                                config_dir=args.llm_config_dir,
                                clip_patch=args.clip_patch
                                )

    print("Load model from {}...".format(args.checkpoint))
    model.load_state_dict(torch.load(args.checkpoint, map_location=get_device(0))['state_dict'])
    print("Load model from {}...  Done".format(args.checkpoint))

    make_preds_batch(args.annotations, model, args.out_results_file, tokenizer, args=args)


if __name__ == '__main__':
    main()
