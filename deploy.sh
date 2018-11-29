#/bin/bash

DESKTOP=/data3/liuliang

MODEL_ROOT=$DESKTOP/pytorch-deep-image-matting
DATA_ROOT=$DESKTOP/data/deep_image_matting/Test


python core/deploy.py \
	--size_h=320 \
	--size_w=320 \
	--imgDir=$DATA_ROOT/comp/image \
    --trimapDir=$DATA_ROOT/comp/trimap \
	--saveDir=$MODEL_ROOT/result/tmp \
    --alphaDir=$DATA_ROOT/comp/alpha \
	--resume=$MODEL_ROOT/model/stage0/ckpt_e1.pth \
	--cuda \
    --stage=0 \
    --arch=vgg16_nobn \
    --crop_or_resize=crop
    #--in_chan=3
    #--arch=resnet50_aspp \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_e300 \
        #--not_strict \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_add_shen_e30 \
	#--trimapDir=$DATA_ROOT/test/trimap \
	#--imgDir=$DATA_ROOT/input_lowers \
	#--trimapDir=$DATA_ROOT/trimap_lowers/Trimap1 \
	#--alphaDir=$DATA_ROOT/alpha \
