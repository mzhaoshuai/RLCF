#!/bin/bash
# Cross domain evalutation, COCO --> Nocaps
export CUDA_VISIBLE_DEVICES=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

# split=$1
split=val_417_in-domain
# split=val_2670_near-domain
# split=val_1413_out-domain
dataset_root=${root}/dataset/nocaps
images_root=${root}/dataset/nocaps/val
annotations=${dataset_root}/nocaps_${split}.json

clip_model_type=ViT-B/16
# clip_model_type=ViT-L/14

# LLM config
llm_config_dir=${root}/pretrained/opt-125m
download_root=${root}/pretrained/clip
subfolder=capdec_opt125m_transformer_coco_01
# subfolder=clipcap_opt125m_transformer_coco_01
# subfolder=clipcap_opt125m_transformer_coco_02
# subfolder=capdec_opt125m_transformer_coco_02

num=no-tta-01
out_results_file=${root}/output/${subfolder}/coco2nocaps_${split}_${num}.json
out_clipscore_file=${root}/output/${subfolder}/coco2nocaps_clips_${split}_${num}.json
checkpoint=${root}/output/${subfolder}/ckpt-latest.pt

runfile=${root}/code/RLCF/caption/predictions.py

python ${runfile}   --download_root ${download_root} \
                    --images_root ${images_root} \
                    --checkpoint ${checkpoint} \
                    --out_results_file ${out_results_file} \
                    --out_clipscore_file ${out_clipscore_file} \
                    --llm_config_dir ${llm_config_dir} \
                    --annotations ${annotations} \
                    --normalize_prefix \
                    --dataset_mode 2 \
                    --clip_model_type ${clip_model_type}

wait


python ${root}/code/RLCF/clipscore/clipscore.py \
        ${out_clipscore_file} ${root}/dataset/nocaps/val --references_json ${dataset_root}/nocaps_${split}_clipscore.json

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
