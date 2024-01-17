# coding=utf-8
import os
import csv
import json
import pickle
import argparse
import torch
from tqdm import tqdm
from PIL import Image, ImageFile
from image_llm.clip import clip
from image_llm.datasets import COCOCapDatasetForEmbedding


# ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda:0')


@torch.no_grad()
def extract_clip_embedding(clip_model_type, out_path, annotations_path, images_path=None, data_mode=0, download_root=None,
                            batch_size=1024):
    """use LAVIS splits
    https://github.com/salesforce/LAVIS/blob/1a0c1e37cf4ec783aa9bcb35bc58e50e9a7e777c/lavis/tasks/captioning.py#L109
    https://github.com/salesforce/LAVIS/blob/main/lavis/configs/datasets/coco/defaults_cap.yaml

    LAVIS train Format:
    [{"caption": "A woman eating fresh vegetables from a bowl.", "image": "val2014/COCO_val2014_000000328757.jpg", "image_id": "coco_328757"}]

    CapDec Format:
    [{"image_id": 296759, "caption": "A store features many types of teddy bears with various outfits. ", "id": 192959}]
    """
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False, download_root=download_root)
    clip_model = clip_model.float()

    with open(annotations_path, 'r') as f:
        data = json.load(f)
    print("%0d samples loaded from json " % len(data))

    all_image_embeddings = []
    all_text_embeddings = []
    all_captions = []
    all_image_ids = []
    text_index = []
    image_index = []
    image_index_cnt = 0
    text_index_cnt = 0
    not_found = 0
    curr_image_batch = []
    curr_text_batch = []
    image_id_to_index = dict()

    for i in tqdm(range(len(data)), ncols=100):
        entry = data[i]
        image_id = int(entry['image_id'].split('_')[-1])

        # extract image embedding,
        if image_id not in image_id_to_index.keys():
            filename = os.path.join(images_path, entry['image'])
            try:
                image = Image.open(filename).convert("RGB")
                curr_image_batch.append(preprocess(image))

                all_image_ids.append(image_id)
                # use image id to identify image
                image_id_to_index[image_id] = image_index_cnt
                image_index_cnt += 1

            except:
                not_found += 1
                print("{} not found".format(filename))
                continue

        # extract image embedding, batch operations
        if len(curr_image_batch) == batch_size or i == len(data) - 1:
            image = torch.stack(curr_image_batch, dim=0).to(device)
            # all_token > 0 will return all tokens
            all_image_embeddings.append(clip_model.encode_image(image, cls=not args.all_token).cpu())
            curr_image_batch = []

        # extract text embedding
        curr_text_batch.append(entry['caption'])
        all_captions.append(entry['caption'])
        image_index.append(image_id_to_index[image_id])
        text_index.append(text_index_cnt)
        text_index_cnt += 1
        # batch operations
        if len(curr_text_batch) == batch_size or i == len(data) - 1:
            caption_tokens = clip.tokenize(curr_text_batch).to(device)
            # it is better to avoid normaliztion in this stage so it will be possible to normelise or not later
            caption_embedding = clip_model.encode_text(caption_tokens, eot=not args.all_token).cpu()
            all_text_embeddings.append(caption_embedding)
            curr_text_batch = []

    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_text_embedding": all_text_embeddings,
                        "clip_image_embedding": all_image_embeddings,
                        "captions": all_captions,
                        'image_ids': all_image_ids,
                        "text_index": text_index,
                        "image_index": image_index},
                        f
                    )

    print("%0d text embeddings saved " % len(all_text_embeddings))
    print("%0d image embeddings saved " % len(all_image_embeddings))
    print(f'not found images = {not_found}')


@torch.no_grad()
def extract_clip_embedding_batch(clip_model_type, out_path, annotations_path, images_path=None, data_mode=0, download_root=None,
                                    batch_size=512):
    """this function will save multiple same image embeddings"""
    # https://discuss.pytorch.org/t/oserror-image-file-is-truncated-150-bytes-not-processed/64445
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False, download_root=download_root)
    clip_model.float()
    dataset = COCOCapDatasetForEmbedding(annotations_path, images_path, preprocess)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=16)

    all_image_embeddings = []
    all_text_embeddings = []
    all_captions = []
    all_image_ids = []

    for i, sample in enumerate(tqdm(train_dataloader, ncols=100)):
        image, captions, image_ids = sample['image'].to(device), sample['caption'], sample['image_id']
        all_captions.extend(captions)
        all_image_ids.extend(image_ids)

        # extract image embedding
        all_image_embeddings.append(clip_model.encode_image(image).cpu())

        # extract text embedding
        caption_tokens = clip.tokenize(captions, truncate=True).to(device)
        caption_embedding = clip_model.encode_text(caption_tokens).cpu()
        all_text_embeddings.append(caption_embedding)

    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)

    text_index = list(range(len(all_captions)))
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_text_embedding": all_text_embeddings,
                        "clip_image_embedding": all_image_embeddings,
                        "captions": all_captions,
                        'image_ids': all_image_ids,
                        "text_index": text_index,
                        "image_index": text_index},
                        f
                    )

    print("%0d text embeddings saved " % len(all_text_embeddings))
    print("%0d image embeddings saved " % len(all_image_embeddings))


@torch.no_grad()
def extract_clip_embedding_txt(clip_model_type, out_path, annotations_path, images_path=None, data_mode=0, download_root=None,
                                    batch_size=512):
    """read captions from a txt file and extract its CLIP text embedding"""
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False, download_root=download_root)
    clip_model = clip_model.float()
    clip_model.eval()

    # read captions from txt file
    if args.dataset_type == 'COCO':
        with open(annotations_path) as f:
            content = [' '.join(x.strip().split(' ')[1:]) for x in f.readlines()]
        all_captions = [x for x in content if len(x) > 1]
    elif args.dataset_type == 'CC12M':
        all_captions = []
        with open(annotations_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row[1]) > 1:
                    all_captions.append(row[1])

    print("The number of captions is {}".format(len(all_captions)))

    all_text_embeddings = []
    for i in tqdm(range(0, len(all_captions), batch_size), ncols=100):
        end = min(i + batch_size, len(all_captions))
        captions = all_captions[i : end]

        # extract text embedding
        caption_tokens = clip.tokenize(captions, truncate=True).to(device)
        caption_embedding = clip_model.encode_text(caption_tokens).cpu()
        all_text_embeddings.append(caption_embedding)

    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    text_index = list(range(len(all_captions)))
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_text_embedding": all_text_embeddings,
                        "clip_image_embedding": None,
                        "captions": all_captions,
                        'image_ids': text_index,
                        "text_index": text_index,
                        "image_index": text_index},
                        f
                    )

    print("%0d text embeddings saved " % len(all_text_embeddings))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'))
    # 0 for COCO!!, 1 for flicker30, 2 humor style,3 romantic,4 factual of style,6 harrypotter, 7 for news.
    parser.add_argument('--dataset_mode', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='COCO', choices=['COCO', 'CC12M'])
    # 1 for both genders, 2 for man only, 3 for woman only 
    parser.add_argument('--fix_gender_imbalance_mode', type=int, default=0)
    parser.add_argument('--out_path', type=str, default=None, help="output file name")
    parser.add_argument('--annotations_path', type=str, default=None, help="annotations location")
    parser.add_argument('--images_path', type=str, default=None, help="images location")
    parser.add_argument('--download_root', type=str, default=None, help="clip download root")
    parser.add_argument('--extract_method', type=int, default=0)
    parser.add_argument('--all_token', type=int, default=0)

    args = parser.parse_args()
    print(args)

    if args.extract_method == 0:
        extract_clip_embedding(args.clip_model_type, args.out_path, args.annotations_path, args.images_path,
                                    download_root=args.download_root)
    elif args.extract_method == 1:
        extract_clip_embedding_batch(args.clip_model_type, args.out_path, args.annotations_path, args.images_path,
                                        download_root=args.download_root)
    elif args.extract_method == 2:
        extract_clip_embedding_txt(args.clip_model_type, args.out_path, args.annotations_path, args.images_path,
                                        download_root=args.download_root)
    else:
        raise NotImplementedError
