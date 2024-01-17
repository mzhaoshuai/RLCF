# coding=utf-8
# extract features and save to a lmdb file
import io
import os
import json


def convert_flickr30k_to_coco_eval_gt(annotations):
    """
    original:
    {"image": "flickr30k-images/97234558.jpg",
    "caption": ["A young girl wades out into the water wearing safety floatation devices on her arms.",
    "The baby walks on the gold and sandy-colored beach with water splashing behind him.",
    "A girl at the shore of a beach with a mountain in the distance.",
    "Little girl in arm floaties exploring the coast line.",
    "A child playing in the ocean."]
    }

    target:
    {"annotations": [{"image_id": 391895, "caption": "A man with a red helmet on a small moped on a dirt road. ", "id": 770337},]
    "images": [{"id": 391895}, {"id": 60623}, {"id": 483108},]
    """
    basedir = os.path.basename(annotations)
    output_dict = {"annotations": [],
                    "images": [],}
    with open(annotations, 'r') as f:
        data = json.load(f)

    for item in data:
        image_id = int(item["image"].split("/")[-1][:-4])
        output_dict["images"].append({"id": image_id})
        for cap in item["caption"]:
            output_dict["annotations"].append({"image_id": image_id, "caption": cap, "id": image_id})

    output_filename = annotations.replace(".json", "_gt.json")
    print("save results to {}".format(output_filename))
    with open(output_filename, 'w') as outfile:
        json.dump(output_dict, outfile)


def convert_flickr30k_to_clipscore_eval_gt(annotations):
    """
    original:
    {"image": "flickr30k-images/97234558.jpg",
    "caption": ["A young girl wades out into the water wearing safety floatation devices on her arms.",
    "The baby walks on the gold and sandy-colored beach with water splashing behind him.",
    "A girl at the shore of a beach with a mountain in the distance.",
    "Little girl in arm floaties exploring the coast line.",
    "A child playing in the ocean."]
    }

    target:
    {"image1": ["two cats are sleeping next to each other.",
    "a grey cat is cuddling with an orange cat on a blanket.", "the orange cat is happy that the black cat is close to it."],
    "image2": ["a dog is wearing ear muffs as it lies on a carpet.", "a black dog and an orange cat are looking at the photographer.",
    "headphones are placed on a dogs ears."]}
    """
    output_dict = dict()
    with open(annotations, 'r') as f:
        data = json.load(f)

    for item in data:
        filename = os.path.basename(item["image"])
        output_dict[filename] = item["caption"]

    output_filename = annotations.replace(".json", "_clips_gt.json")
    print("save results to {}".format(output_filename))
    with open(output_filename, 'w') as outfile:
        json.dump(output_dict, outfile)


def convert_coco_to_clipscore_eval_gt(annotations):
    convert_flickr30k_to_clipscore_eval_gt(annotations)


if __name__ == "__main__":
    # for flickr dataset
    # annotations = "flickr30k/annotations/flickr30k_test.json"
    # convert_flickr30k_to_coco_eval_gt(annotations)
    # convert_flickr30k_to_clipscore_eval_gt(annotations)

    # for coco dataset
    annotations = "coco2014/coco_karpathy_test.json"
    convert_coco_to_clipscore_eval_gt(annotations)
