#!/bin/bash
gpu=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################
data_root=${root}/dataset/tta_data

# Config
arch=ViT-B/16
coop_weight=${root}/unified_lv_env/coop/coop_16shots_nctx4_cscFalse_ctpend_vitb16_seed1
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

# momentum update
momentum_update=0
update_freq=256
tta_momentum=0.9999
update_w=1.0


runfile=${root}/code/RLCF/TPT/tune_cls_tpt.py

for num in 01
do
    case ${num} in
        01 )
            tta_steps=3
            lr=1e-5
            testsets='A/V/R/I/K'
            ;;
        * )
            ;;
    esac

output=${root}/output/tune_tpt_cls_tpt_${num}

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
        --update_w ${update_w} \
        --sample_k ${sample_k}

done

echo "Finished!!!"
echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
