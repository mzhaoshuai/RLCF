#!/bin/bash
# Cross domain evalutation, COCO --> flickr30k
export CUDA_VISIBLE_DEVICES=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

dataset_mode=1
split=test
dataset_root=${root}/dataset/flickr30k
images_root=${root}/dataset/flickr30k
annotations=${dataset_root}/annotations/flickr30k_${split}.json

# LLM config
llm_config_dir=${root}/pretrained/opt-125m
download_root=${root}/pretrained/clip
subfolder=capdec_opt125m_transformer_coco_01
# subfolder=clipcap_opt125m_transformer_coco_01

num=01
out_results_file=${root}/output/${subfolder}/coco2flickr_${split}_${num}.json
out_clipscore_file=${root}/output/${subfolder}/coco2flickr_clips_${split}_${num}.json
checkpoint=${root}/output/${subfolder}/ckpt-latest.pt


runfile=${root}/RLCF/caption/predictions.py

python ${runfile}   --download_root ${download_root} \
                    --images_root ${images_root} \
                    --checkpoint ${checkpoint} \
                    --out_results_file ${out_results_file} \
                    --out_clipscore_file ${out_clipscore_file} \
                    --llm_config_dir ${llm_config_dir} \
                    --annotations ${annotations} \
                    --normalize_prefix \
                    --dataset_mode ${dataset_mode}

wait

python ${root}/RLCF/clipscore/clipscore.py \
        ${out_clipscore_file} ${dataset_root}/flickr30k-images --references_json ${dataset_root}/annotations/flickr30k_test_clips_gt.json

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
