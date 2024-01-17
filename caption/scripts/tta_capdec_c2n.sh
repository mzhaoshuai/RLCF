#!/bin/bash
# Cross domain TTA, COCO --> Nocaps
export CUDA_VISIBLE_DEVICEDS=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

dataset_root=${root}/dataset/nocaps
images_root=${root}/dataset/nocaps/val
split=val_417_in-domain
split=val_2670_near-domain
split=val_1413_out-domain
suffix=tta

# LLM config
llm_config_dir=${root}/pretrained/opt-125m
download_root=${root}/pretrained/clip
subfolder=capdec_opt125m_transformer_coco_01

# TTA learning strategy
tta_steps=3
tta_lr=3e-6
tta_weight_decay=0.0

clip_model_type=ViT-B/16
reward_arch=ViT-L/14
reward_amplify=0
reward_process=1
process_batch=0
sample_k=6
multiple_reward_models=0

# momentum update
momentum_update=0
update_freq=64
tta_momentum=0.9998
update_w=1


runfile=${root}/code/RLCF/caption/capdec_tta.py


for num in 01-0
do
    case ${num} in
        01-0 )
            tta_steps=4
            tta_lr=3e-6
            sample_k=6
            split=val_417_in-domain
            ;;
        01-1 )
            split=val_2670_near-domain
            ;;
        01-2 )
            split=val_1413_out-domain
            ;;
        * )
            ;;
    esac

checkpoint=${root}/output/${subfolder}/ckpt-latest.pt
annotations=${dataset_root}/nocaps_${split}.json

out_results_file=${root}/output/${subfolder}/c2n_${split}_${suffix}_${num}.json
out_clipscore_file=${root}/output/${subfolder}/c2n_clips_${split}_${suffix}_${num}.json


python ${runfile}   --download_root ${download_root} \
                    --images_root ${images_root} \
                    --checkpoint ${checkpoint} \
                    --out_results_file ${out_results_file} \
                    --out_clipscore_file ${out_clipscore_file} \
                    --llm_config_dir ${llm_config_dir} \
                    --annotations ${annotations} \
                    --normalize_prefix \
                    --tta_steps ${tta_steps} \
                    --tta_lr ${tta_lr} \
                    --tta_weight_decay ${tta_weight_decay} \
                    --reward_arch ${reward_arch} \
                    --reward_amplify ${reward_amplify} \
                    --reward_process ${reward_process} \
                    --process_batch ${process_batch} \
                    --momentum_update ${momentum_update} \
                    --update_freq ${update_freq} \
                    --tta_momentum ${tta_momentum} \
                    --update_w ${update_w} \
                    --sample_k ${sample_k} \
                    --multiple_reward_models ${multiple_reward_models} \
                    --clip_model_type ${clip_model_type} \
                    --dataset_mode 2

wait


python ${root}/code/RLCF/clipscore/clipscore.py \
        ${out_clipscore_file} ${root}/dataset/nocaps/val --references_json ${dataset_root}/nocaps_${split}_clipscore.json

done

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
