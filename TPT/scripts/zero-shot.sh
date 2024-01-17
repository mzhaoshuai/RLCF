#!/bin/bash
gpu=$1

root=/home/shuzhao/Data
data_root=${root}/dataset/tta_data
testsets='A/V/R/I/K'

coop_weight=${root}/pretrained/coop/coop_16shots_nctx4_cscFalse_ctpend_vitb16_seed1
arch=ViT-B/16
ctx_init=a_photo_of_a
tta_steps=0
lr=5e-3
# augmentation views and selection ratio
batch_size=64
selection_p=0.1

runfile=${root}/code/RLCF/TPT/zero_shot.py


# for testsets in "Caltech101" "Cars" "Pets" "Flower102" "Aircraft" "Food101" "DTD" "eurosat" "UCF101" "SUN397"
for num in 01
do
	case ${num} in
        01 )
            testsets='A/V/R/I/K'
            arch=RN50
            coop_weight=None
            resolution=224
			;;
        02 )
            testsets='A/V/R/I/K'
            arch=RN50x4
            coop_weight=None
            resolution=384
			;;
		* )
            arch=ViT-L/14
            coop_weight=None
            resolution=224
            batch_size=128
			;;
	esac

output=${root}/output/zero_shot_${num}

python ${runfile} ${data_root} \
        --test_sets ${testsets} \
        -a ${arch} \
        --batch_size ${batch_size} \
        --gpu ${gpu} \
        --ctx_init ${ctx_init} \
        --tta_steps ${tta_steps} \
        --lr ${lr} \
        --output ${output} \
        --resolution ${resolution} \
        --load ${coop_weight}

done

echo "Finished!!!"
echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
