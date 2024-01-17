#!/bin/bash
gpu=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

# Dataset
data_root=${root}/dataset/tta_data

# Config
arch=ViT-B/16
coop_weight=${root}/pretrained/coop/coop_16shots_nctx4_cscFalse_ctpend_vitb16_seed1
testsets='A/V/R'
ctx_init=a_photo_of_a
tta_steps=3
lr=1e-5
weight_decay=5e-4

# augmentation views and selection ratio
batch_size=64
selection_p=0.1

# Config for CLIP reward
reward_arch=ViT-L/14
reward_amplify=0
reward_process=1
process_batch=0
sample_k=3

# momentum update
momentum_update=0
update_freq=256
tta_momentum=0.9999
update_w=1.0
multiple_reward_models=0

# KD loss choices, ["ATKD", "KD", "DKD"]
kd_loss="KD"


runfile=${root}/code/RLCF/TPT/tune_cls_kd.py


for num in 01
do
    case ${num} in
        01 )
            tta_steps=3
            lr=1e-5
            testsets='A/V/R/I/K'
            reward_arch=ViT-L/14
            sample_k=3
            ;;
        02 )
            tta_steps=3
            lr=1e-5
            testsets='A/V/R/I/K'
            reward_arch=ViT-L/14
            sample_k=3
            kd_loss="ATKD"
            ;;
        * )
            ;;
    esac

output=${root}/output/finetune_cls_kd_${num}

python ${runfile} ${data_root} \
        --test_sets ${testsets} \
        -a ${arch} \
        --batch_size ${batch_size} \
        --selection_p ${selection_p} \
        --gpu ${gpu} \
        --tpt \
        --ctx_init ${ctx_init} \
        --tta_steps ${tta_steps} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --output ${output} \
        --load ${coop_weight} \
        --reward_arch ${reward_arch} \
        --reward_amplify ${reward_amplify} \
        --reward_process ${reward_process} \
        --process_batch ${process_batch} \
        --momentum_update ${momentum_update} \
        --update_freq ${update_freq} \
        --tta_momentum ${tta_momentum} \
        --sample_k ${sample_k} \
        --update_w ${update_w} \
        --multiple_reward_models ${multiple_reward_models} \
        --kd_loss ${kd_loss}

done

echo "Finished!!!"

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
