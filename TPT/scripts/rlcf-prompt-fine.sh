#!/bin/bash
gpu=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

# Dataset
data_root=${root}/dataset/finegrained-dataset

# Config
arch=ViT-B/16
coop_weight=${root}/pretrained/coop/coop_16shots_nctx4_cscFalse_ctpend_vitb16_seed1
testsets='A/V/R'
ctx_init=a_photo_of_a
tta_steps=3
lr=5e-3
weight_decay=5e-4

# augmentation views and selection ratio
batch_size=64
selection_p=0.1

reward_arch=ViT-L/14
reward_amplify=0
reward_process=1
process_batch=0
sample_k=3


runfile=${root}/code/RLCF/TPT/tpt_cls_rl.py


num=01
for testsets in "Caltech101" "Cars" "Pets" "Flower102" "Aircraft" "Food101" "SUN397" "DTD" "eurosat" "UCF101"
do
    case ${num} in
        01 )
            tta_steps=5
            lr=7e-3
            reward_arch=ViT-L/14
            sample_k=3
            ;;
        * )
            ;;
    esac

output=${root}/output/rlcf_prompt_fine_${num}

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
        --reward_amplify ${reward_amplify} \
        --reward_process ${reward_process} \
        --process_batch ${process_batch} \
        --reward_arch ${reward_arch} \
        --sample_k ${sample_k} \
        --augmix ${augmix}

done

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
