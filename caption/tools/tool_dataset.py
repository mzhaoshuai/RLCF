# coding=utf-8
# coding=utf-8
import os
import lmdb
import json
import torch
from PIL import Image


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
                    "image_subpath": entry["image"]
                    }



def  subsample_test(annotations: str, output_file: str):
    with open(annotations, 'r') as f:
        data = json.load(f)    
