#!/bin/bash
# Evalutation on COCO
export CUDA_VISIBLE_DEVICES=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

split=test
dataset_root=${root}/dataset/coco2014
images_root=${root}/dataset/coco2014
annotations=${dataset_root}/coco_karpathy_${split}.json

clip_model_type=ViT-B/16
# clip_model_type=ViT-L/14

# LLM config
llm_config_dir=${root}/pretrained/opt-125m
download_root=${root}/pretrained/clip
subfolder=capdec_opt125m_transformer_coco_01
# subfolder=clipcap_opt125m_transformer_coco_01
# subfolder=clipcap_opt125m_transformer_coco_02

num=01
out_results_file=${root}/output/${subfolder}/coco_${split}_${num}.json
out_clipscore_file=${root}/output/${subfolder}/coco_clips_${split}_${num}.json
checkpoint=${root}/output/${subfolder}/ckpt-latest.pt


runfile=${root}/code/RLCF/caption/predictions.py

python ${runfile}  --download_root ${download_root} \
                    --images_root ${images_root} \
                    --checkpoint ${checkpoint} \
                    --out_results_file ${out_results_file} \
                    --out_clipscore_file ${out_clipscore_file} \
                    --llm_config_dir ${llm_config_dir} \
                    --annotations ${annotations} \
                    --normalize_prefix \
                    --clip_model_type ${clip_model_type}

wait

python ${root}/code/RLCF/clipscore/clipscore.py \
        ${out_clipscore_file} ${dataset_root}/val2014 --references_json ${dataset_root}/coco_karpathy_${split}_clips_gt.json

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
