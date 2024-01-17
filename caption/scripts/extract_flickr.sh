#!/bin/bash
# used for the training of CLIPCap and CapDec

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################
dataset_root=${root}/dataset/flickr30k

images_path=${dataset_root}
download_root=${root}/pretrained/clip
annotations_path=${dataset_root}/annotations/flickr30k_train.json

extract_method=1
out_path=${dataset_root}/flickr_train_set_image_text_vitb16_v2.pkl

runfile=${root}/RLCF/caption/extractor_pickle.py

python ${runfile} --clip_model_type ViT-B/16 \
                    --out_path ${out_path} \
                    --annotations_path ${annotations_path} \
                    --images_path ${images_path} \
                    --download_root ${download_root} \
                    --extract_method ${extract_method}

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
