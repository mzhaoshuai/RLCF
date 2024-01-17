#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
# used for the training CLIPCap

### Remember to change root to /YOUR/PATH ###
root=/home/shuzhao/Data
#############################################

dataset_root=${root}/dataset/coco2014
split=train

clip_model_type=ViT-B/16
download_root=${root}/pretrained/clip
data=${dataset_root}/COCO_train_set_image_text_vitb16_v2.pkl
llm_config_dir=${root}/pretrained/opt-125m

# learning strategy
epochs=10
lr=2e-5
bs=40
cap_model=CLIPCap
mapping_type=transformer
runfile=${root}/code/RLCF/caption/train.py


for num in 01
do
    case ${num} in
        01 )
            ;;
        02 )
            clip_model_type=ViT-L/14
            data=${dataset_root}/COCO_train_set_image_text_vitl14.pkl
            ;;
        * )
            ;;
    esac

output=${root}/output/clipcap_opt125m_${mapping_type}_coco_${num}


python ${runfile}   --clip_model_type ${clip_model_type} \
                    --download_root ${download_root} \
                    --cap_model ${cap_model} \
                    --data ${data} \
                    --out_dir ${output} \
                    --noise_variance 0.016 \
                    --llm_config_dir ${llm_config_dir} \
                    --mapping_type ${mapping_type} \
                    --lr ${lr} \
                    --bs ${bs} \
                    --epochs ${epochs} \
                    --normalize_prefix \
                    --use_image_embedding


done

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
