#!/bin/bash
gpu=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################
data_root=${root}/dataset/tta_data


testsets='A/V/R/K'
arch=ViT-B/16
batch_size=64
ctx_init=a_photo_of_a
tta_steps=0

num=01
runfile=${root}/code/RLCF/TPT/tpt_cls.py
output=${root}/output/tpt_prompt_${num}


python ${runfile} ${data_root} \
        --test_sets ${testsets} \
        -a ${arch} \
        --batch_size ${batch_size} \
        --gpu ${gpu} \
        --tpt \
        --ctx_init ${ctx_init} \
        --tta_steps ${tta_steps} \
        --output ${output}

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
