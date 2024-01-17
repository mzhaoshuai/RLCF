# coding=utf-8
# extract features and save to a lmdb file
import io
import os
import lmdb
import pickle
import argparse
import torch
from tqdm import tqdm

from PIL import ImageFile
from image_llm import clip
from image_llm.datasets import COCOCapDatasetForEmbedding

device = torch.device('cuda:0')

ImageFile.LOAD_TRUNCATED_IMAGES = True


@torch.no_grad()
def extract_clip_embedding_batch(clip_model_type, data_dir, annotations_path, images_path, download_root=None,
                                    batch_size=128, write_frequency=500, args=None):
    """this function will save multiple same image embeddings"""

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False, download_root=download_root)
    clip_model.eval()
    clip_model = clip_model.float()

    dataset = COCOCapDatasetForEmbedding(annotations_path, images_path, preprocess)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    # where lmdb locate
    if not os.path.exists(os.path.join(data_dir, "{}_lmdb".format(args.lmdb_name))):
        os.makedirs(os.path.join(data_dir, "{}_lmdb".format(args.lmdb_name)), exist_ok=True)	
    lmdb_path = os.path.join(data_dir, "{}_lmdb".format(args.lmdb_name), "{}.lmdb".format(args.lmdb_name))    
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1 << 43,
                   readonly=False,
                   meminit=False,
                   map_async=True)

    used_image_ids = set()
    txn = db.begin(write=True)
    for i, sample in enumerate(tqdm(train_dataloader, ncols=100)):
        image, captions, image_ids = sample['image'].to(device), sample['caption'], sample['image_id']

        # extract image embedding
        image_embeddings = clip_model.encode_image(image, cls=args.image_only_cls).cpu()

        # extract text embedding
        caption_tokens = clip.tokenize(captions).to(device)
        caption_embedding = clip_model.encode_text(caption_tokens, eot=args.text_only_eot).cpu()

        for j, idx in enumerate(image_ids):
            # save image
            if idx not in used_image_ids:
                image_key = "image_" + idx
                # flag = txn.put(image_key.encode(), pickle.dumps(image_embeddings[j, ...]))
                flag = txn.put(image_key.encode(), image_embeddings[j, ...].numpy().tobytes())
                if not flag: raise IOError("LMDB write error!")
                used_image_ids.add(idx)

            # save text
            text_key = "text_" + idx
            # flag = txn.put(text_key.encode(), pickle.dumps(caption_embedding[j, ...]))
            flag = txn.put(text_key.encode(), caption_embedding[j, ...].numpy().tobytes())
            if not flag: raise IOError("LMDB write error!")        

        # write after certain samples
        if i % write_frequency == 0:
            # print("[{}/{}]".format(i, smaple_num))
            txn.commit()
            txn = db.begin(write=True)

    print("image_embeddings shape {}".format(image_embeddings.shape))
    print("caption_embedding shape {}".format(caption_embedding.shape))

    # finish iterating through dataset
    txn.commit()
    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'))
    # 0 for COCO!!, 1 for flicker30, 2 humor style,3 romantic,4 factual of style,6 harrypotter, 7 for news.
    parser.add_argument('--dataset_mode', type=int, default=0)
    # 1 for both genders, 2 for man only, 3 for woman only 
    parser.add_argument('--fix_gender_imbalance_mode', type=int, default=0)
    parser.add_argument('--out_path', type=str, default=None, help="output file name")
    parser.add_argument('--annotations_path', type=str, default=None, help="annotations location")
    parser.add_argument('--images_path', type=str, default=None, help="images location")
    parser.add_argument('--download_root', type=str, default=None, help="clip download root")
    parser.add_argument('--image_only_cls', type=int, default=1)
    parser.add_argument('--text_only_eot', type=int, default=1)
    parser.add_argument('--lmdb_name', type=str, default='clip', help="the name of the lmdb file")

    args = parser.parse_args()
    print(args)

    extract_clip_embedding_batch(args.clip_model_type, args.out_path, args.annotations_path, args.images_path,
                                    download_root=args.download_root, args=args)
