#!/bin/bash
# used for the training of CLIPCap and CapDec
export CUDA_VISIBLE_DEVICES=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################
dataset_root=${root}/dataset/coco2014

dataset_mode=0
dataset_type=COCO
images_path=${dataset_root}
download_root=${root}/pretrained/clip
annotations_path=${dataset_root}/coco_karpathy_train.json

# Config
all_token=0
extract_method=0
# clip_model_type=ViT-B/16
# out_path=${dataset_root}/COCO_train_set_image_text_vitb16_v2.pkl
clip_model_type=ViT-L/14
out_path=${dataset_root}/COCO_train_set_image_text_vitl14.pkl


runfile=${root}/RLCF/caption/extractor_pickle.py

python ${runfile} --clip_model_type ${clip_model_type} \
                    --dataset_mode ${dataset_mode} \
                    --dataset_type ${dataset_type} \
                    --out_path ${out_path} \
                    --annotations_path ${annotations_path} \
                    --images_path ${images_path} \
                    --download_root ${download_root} \
                    --all_token ${all_token} \
                    --extract_method ${extract_method}

# 566746 text embeddings saved 
# 113286 image embeddings saved 
# not found images = 1
echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
