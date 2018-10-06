#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-deep-image-matting
DATA_ROOT=/home/liuliang/Desktop/matting_data
#DATA_ROOT=/home/liuliang/Desktop/dataset_shen_matting

python core/deploy.py \
	--size_h=480 \
	--size_w=480 \
	--imgDir=$DATA_ROOT/test/image \
	--saveDir=$ROOT/result/tmp \
	--alphaDir=$DATA_ROOT/test/alpha \
	--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_add_shen_e30 \
	--resume=$ROOT/model/input_480_stage3/ckpt_e40.pth \
	--cuda \
        --stage=1 \
        --arch=vgg16 \
        #--in_chan=3
        #--arch=resnet50_aspp \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_e300 \
        #--not_strict \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_deeplabv3+_e500 \
	#--trimapDir=$DATA_ROOT/test/trimap \
