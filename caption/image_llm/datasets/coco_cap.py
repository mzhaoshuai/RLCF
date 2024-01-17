# coding=utf-8
import os
import sys
import json
import pickle
import torch
from PIL import Image
from typing import Tuple
from transformers import GPT2Tokenizer, AutoTokenizer


class COCOCLIPCapTrainDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: str,  prefix_length: int, config_dir: str = "gpt2",
                    normalize_prefix=False, use_image_embedding=False, force_gen_tokens=True):
        # self.tokenizer = GPT2Tokenizer.from_pretrained(config_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(config_dir)
        token_filename = os.path.basename(config_dir) + '_' + os.path.basename(data_path).split('.')[0]

        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        # load embeddings
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        self.prefixes = all_data["clip_image_embedding"] if use_image_embedding else all_data["clip_text_embedding"]
        self.image_ids = all_data["image_ids"]
        self.captions = all_data["captions"]
        self.text_index = all_data["text_index"]
        self.image_index = all_data["image_index"]
        print("The dataset size is {}".format(len(self.captions)))

        token_filename = "{}_image_tokens.pkl".format(token_filename) if use_image_embedding else \
                            "{}_text_tokens.pkl".format(token_filename)
        llm_token_file = os.path.join(os.path.dirname(data_path), token_filename)
        if os.path.isfile(llm_token_file) and not force_gen_tokens:
            # read token file
            print(f"[dataset] loading {token_filename}...")
            with open(llm_token_file, 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
            print(f"[dataset] loading {token_filename}... Done")
        else:
            # if not, create the token file
            print(f"[dataset] creating {token_filename}...")
            self.captions_tokens, self.caption2embedding = [], []
            max_seq_len = 0
            for i, caption in enumerate(self.captions):
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
                self.caption2embedding.append(self.image_index[i] if use_image_embedding  else self.text_index[i])

                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])

            with open(llm_token_file, 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f, protocol=4)
            print(f"[dataset] creating {token_filename}...Done")

        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens

        # mask is zero where we out of sequence
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask

        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]

        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        return {
                "tokens": tokens,
                "mask": mask,
                "prefix": prefix,
                }


class COCOCapTrainDataset(torch.utils.data.Dataset):

    def __init__(self, annotations: str, images_root: str,
                    prefix_length: int, transforms=None, config_dir: str = "gpt2",
                    token_filename: str = "gpt2"
                    ):
        """
        The dataset will return the image for downstream applications.

        LAVIS train Format:
        [{"caption": "A woman eating fresh vegetables from a bowl.", "image": "val2014/COCO_val2014_000000328757.jpg", "image_id": "coco_328757"}]
        """
        super().__init__()
        with open(annotations, 'r') as f:
            data = json.load(f)
        self.captions = [d["caption"] for d in data]
        self.image_ids = [d["image_id"] for d in data]
        self.image_subpath = [d["image"] for d in data]
        self.transforms = transforms

        self.tokenizer = GPT2Tokenizer.from_pretrained(config_dir)
        self.prefix_length = prefix_length

        llm_token_file = os.path.join(images_root, "{}_tokens.pkl".format(token_filename))
        # raed or create token file
        if os.path.isfile(llm_token_file):
            with open(llm_token_file, 'rb') as f:
                self.captions_tokens, self.caption2imagepath, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2imagepath = []
            max_seq_len = 0
            for i, caption in enumerate(self.captions):
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
                self.caption2imagepath.append(os.path.join(images_root, self.image_subpath[i]))

                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])

            with open(llm_token_file, 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2imagepath, max_seq_len], f)

        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        caption = self.captions[item]
        tokens, mask = self.pad_tokens(item)
        image = Image.open(self.caption2imagepath[item]).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {
                "llm_tokens": tokens,
                "mask": mask,
                "image": image,
                "caption": caption
                }

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens

        # mask is zero where we out of sequence
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask

        return tokens, mask


class COCOCapEvalDataset(torch.utils.data.Dataset):
    def __init__(self, annotations: str, images_root=None, transform=None):
        super().__init__()
        with open(annotations, 'r') as f:
            data = json.load(f)
        self.data = data
        self.images_root = images_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        entry = self.data[item]
        filename = os.path.join(self.images_root, entry["image"])
        image = Image.open(filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


class COCOCapDatasetForEmbedding(torch.utils.data.Dataset):
    def __init__(self, annotations: str, images_root=None, transform=None, split='train'):
        """use LAVIS splits
        https://github.com/salesforce/LAVIS/blob/1a0c1e37cf4ec783aa9bcb35bc58e50e9a7e777c/lavis/tasks/captioning.py#L109
        https://github.com/salesforce/LAVIS/blob/main/lavis/configs/datasets/coco/defaults_cap.yaml

        LAVIS train Format:
        [{"caption": "A woman eating fresh vegetables from a bowl.", "image": "val2014/COCO_val2014_000000328757.jpg", "image_id": "coco_328757"}]

        CapDec Format:
        [{"image_id": 296759, "caption": "A store features many types of teddy bears with various outfits. ", "id": 192959}]
        """
        super().__init__()
        with open(annotations, 'r') as f:
            data = json.load(f)
        self.data = data
        self.images_root = images_root
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        entry = self.data[item]
        filename = os.path.join(self.images_root, entry["image"])
        image = Image.open(filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.split == 'train':
            return {
                    "image": image,
                    "image_id": entry["image_id"],
                    "caption": entry["caption"],
                    "image_subpath": entry["image"]
                    }
        else:
            return {
                    "image": image,
                    "image_id": int(entry["image"].split("_")[-1][:-4]),
                    "caption": entry["caption"][0],
                    "image_subpath": os.path.basename(entry["image"])
                    }          


class Flickr30kCapDatasetForEmbedding(COCOCapDatasetForEmbedding):

    def __getitem__(self, item: int):
        entry = self.data[item]
        filename = os.path.join(self.images_root, entry["image"])
        image = Image.open(filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.split == 'train':
            return {
                    "image": image,
                    "image_id": entry["image_id"],
                    "caption": entry["caption"],
                    "image_subpath": entry["image"]
                    }
        else:
            return {
                    "image": image,
                    "image_id": int(entry["image"].split("/")[-1][:-4]),
                    "caption": entry["caption"][0],
                    "image_subpath": os.path.basename(entry["image"])
                    }


class NocapsCapDatasetForEmbedding(COCOCapDatasetForEmbedding):

    def __getitem__(self, item: int):
        entry = self.data[item]
        filename = os.path.join(self.images_root, entry["image"])
        image = Image.open(filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
                "image": image,
                "image_id": entry["image_id"],
                "caption": entry["caption"][0],
                "image_subpath": os.path.basename(entry["image"])
                }


if __name__ == "__main__":
    import clip
    from tqdm import tqdm
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    device = 'cpu'
    root = "dataset"
    download_root = os.path.join(root, "unified_lv_env")
    annotations = os.path.join(root, "coco2014/coco_karpathy_train.json")

    clip_model, preprocess = clip.load('ViT-B/32', device=device, jit=False, download_root=download_root)
    dataset = COCOCapDatasetForEmbedding(annotations, os.path.dirname(annotations), preprocess)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=8)

    all_captions = 0
    all_images = 0
    for i, sample in enumerate(tqdm(train_dataloader, ncols=100)):
        if i == 0:
            print(sample['caption'])
        all_images += len(sample["image"])
        all_captions += len(sample["caption"])

    print(all_captions, all_images)
