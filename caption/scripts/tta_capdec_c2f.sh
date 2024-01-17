#!/bin/bash
# Cross domain TTA, COCO --> flickr30k
export CUDA_VISIBLE_DEVICEDS=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

split=test
suffix=tta
dataset_root=${root}/dataset/flickr30k
images_root=${root}/dataset/flickr30k
annotations=${dataset_root}/annotations/flickr30k_${split}.json

# LLM config
llm_config_dir=${root}/pretrained/opt-125m
download_root=${root}/pretrained/clip
subfolder=capdec_opt125m_transformer_coco_01
checkpoint=${root}/output/${subfolder}/ckpt-latest.pt

# TTA learning strategy
tta_steps=4
tta_lr=5e-6
tta_weight_decay=0.0

# Config for CLIP reward
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
update_w=1.0


runfile=${root}/RLCF/caption/capdec_tta.py

for num in 01
do
    case ${num} in
        01 )
            tta_steps=4
            tta_lr=5e-6
            sample_k=6
            multiple_reward_models=0
            ;;
        02 )
            tta_steps=4
            tta_lr=5e-6
            sample_k=6
            multiple_reward_models=1
            ;;
        * )
            ;;
    esac

out_results_file=${root}/output/${subfolder}/coco2flickr30k_policy_${split}_${suffix}_${num}.json
out_clipscore_file=${root}/output/${subfolder}/coco2flickr_clips_policy_${split}_${num}.json

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
                    --dataset_mode 1

wait

python ${root}/RLCF/clipscore/clipscore.py \
        ${out_clipscore_file} ${dataset_root}/flickr30k-images --references_json ${dataset_root}/annotations/flickr30k_test_clips_gt.json

done

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
