#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-deep-image-matting
DATA_ROOT=/home/liuliang/Desktop/matting_data
#DATA_ROOT=/home/liuliang/Desktop/dataset_shen_matting

python core/deploy.py \
	--size_h=480 \
	--size_w=480 \
	--imgDir=$DATA_ROOT/test/image \
	--trimapDir=$DATA_ROOT/test/trimap \
	--saveDir=$ROOT/result/tmp \
	--alphaDir=$DATA_ROOT/test/alpha \
	--resume=$ROOT/model/input_480_stage1/ckpt_e40.pth \
	--cuda \
        --stage=1 \
        #--not_strict \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_deeplabv3+_e500 \
