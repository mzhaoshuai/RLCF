#!/bin/bash
export CUDA_VISIBLE_DEVICEDS=$1

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

code_path=${root}/code/RLCF/retrieval
runfile=${code_path}/zero_shot.py

for num in 01
do
    case ${num} in
        * )
            ;;
    esac

output=${root}/output/zeroshot_clip_flickr_ret_${num}

python ${runfile} \
        --cfg-path ${code_path}/lavis/projects/clip/exp_flickr_ret_tta.yaml \
        --output ${output}

output=${root}/output/zeroshot_clip_coco_ret_${num}

python ${runfile} \
        --cfg-path ${code_path}/lavis/projects/clip/exp_coco_ret_tta.yaml \
        --output ${output}

done

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
