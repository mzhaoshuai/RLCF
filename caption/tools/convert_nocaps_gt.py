# coding=utf-8
# extract features and save to a lmdb file
import io
import os
import json


def convert_nocaps_to_coco_eval_gt(annotations):
    """
    the filename of the annotations should be nocaps_val_4500_captions.json
    
    Annotations Format
    {
        "licenses": [],
        "info": {
            "url": "http://nocaps.org",
            "date_created": "2018/11/06",
            "version": "0.1",
            "description": "nocaps validation dataset",
            "contributor": "nocaps team",
            "year": 2018
        },
        "images": [
            {
                "id": 0,
                "open_images_id": "0013ea2087020901",
                "height": 1024,
                "width": 732,
                "coco_url": "https://s3.amazonaws.com/nocaps/val/0013ea2087020901.jpg",
                "file_name": "0013ea2087020901.jpg",
                "license": 0,
                "date_captured": "2018-11-06 11:04:33"
            },
        ]
        "annotations": [  // This field is absent in test set.
            {
                "image_id": 0,
                "id": 0,
                "caption": "A baby is standing in front of a house."
            }
        ]
    }
    """
    all_coco_format = {
                "in-domain" : [],
                "near-domain": [],
                "out-domain": []
    }
    all_clipscore_format = {
                "in-domain" : dict(),
                "near-domain": dict(),
                "out-domain": dict()
    }

    with open(annotations, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations_info = data['annotations']

    # image_id -> captions
    image_id_to_captions = dict()
    for item in annotations_info:
        image_id = item['image_id']
        if image_id not in image_id_to_captions:
            image_id_to_captions[image_id] = [item['caption'],]
        else:
            image_id_to_captions[image_id].append(item['caption'])

    # get image info
    for item in images:
        image_id = item['id']
        coco_format = {'image': item["file_name"],
                        'caption': image_id_to_captions[image_id],
                        'image_id': image_id
                        }

        all_coco_format[item['domain']].append(coco_format)
        all_clipscore_format[item['domain']][item["file_name"]] = image_id_to_captions[image_id]

    # save
    for domain in all_coco_format.keys():
        length = str(len(all_coco_format[domain]))
        output_filename = annotations.replace("nocaps_val_4500_captions.json", "nocaps_val_{}_{}.json".format(length, domain))
        print("save results to {}".format(output_filename))
        with open(output_filename, 'w') as outfile:
            json.dump(all_coco_format[domain], outfile)
        
        # save clipscore reference gt
        output_filename = annotations.replace("nocaps_val_4500_captions.json", "nocaps_val_{}_{}_clipscore.json".format(length, domain))
        print("save results to {}".format(output_filename))
        with open(output_filename, 'w') as outfile:
            json.dump(all_clipscore_format[domain], outfile)


if __name__ == "__main__":
    annotations = "nocaps/nocaps_val_4500_captions.json"
    convert_nocaps_to_coco_eval_gt(annotations)
