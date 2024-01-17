#!/bin/bash
export CUDA_VISIBLE_DEVICEDS=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################
code_path=${root}/code/RLCF/retrieval

# clip_model_type=ViT-B-16
tta_steps=3
lr=1e-6
weight_decay=5e-4

# reward setting
reward_arch=ViT-L-14
reward_amplify=0
reward_process=1
process_batch=0
sample_k_t2i=8
sample_k_i2t=16
multiple_reward_models=0

# momentum update
momentum_update=0
update_freq=64
tta_momentum=0.9998
update_w=1.0


runfile=${code_path}/clip_ret_kd.py
for num in 01
do
    case ${num} in
        01 )
            tta_steps=3
            lr=1e-6
            reward_arch=ViT-L-14
            ;;
        * )
            ;;
    esac

output=${root}/output/tta_clip_flickr_ret_kd_${num}


python ${runfile} \
        --cfg-path ${code_path}/lavis/projects/clip/exp_flickr_ret_tta.yaml \
        --tta_steps ${tta_steps} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --momentum_update ${momentum_update} \
        --update_freq ${update_freq} \
        --tta_momentum ${tta_momentum} \
        --update_w ${update_w} \
        --reward_arch ${reward_arch} \
        --reward_amplify ${reward_amplify} \
        --reward_process ${reward_process} \
        --process_batch ${process_batch} \
        --sample_k ${sample_k_t2i} \
        --output ${output} \
        --multiple_reward_models ${multiple_reward_models} \
        --retrieval_task "text2image"


python ${runfile} \
        --cfg-path ${code_path}/lavis/projects/clip/exp_flickr_ret_tta.yaml \
        --tta_steps ${tta_steps} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --momentum_update ${momentum_update} \
        --update_freq ${update_freq} \
        --tta_momentum ${tta_momentum} \
        --update_w ${update_w} \
        --reward_arch ${reward_arch} \
        --reward_amplify ${reward_amplify} \
        --reward_process ${reward_process} \
        --process_batch ${process_batch} \
        --sample_k ${sample_k_i2t} \
        --output ${output} \
        --multiple_reward_models ${multiple_reward_models} \
        --retrieval_task "image2text"

done

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null

