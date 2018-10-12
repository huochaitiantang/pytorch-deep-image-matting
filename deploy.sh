#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-deep-image-matting
#DATA_ROOT=/home/liuliang/Desktop/matting_data/test
#DATA_ROOT=/home/liuliang/Desktop/dataset_shen_matting
#DATA_ROOT=/home/liuliang/Desktop/dataset_alpha_matting
DATA_ROOT=/home/liuliang/Desktop/dataset_deep_image_matting/Test_set
#DATA_ROOT=/home/liuliang/Desktop/photo_test

python core/deploy.py \
	--size_h=320 \
	--size_w=320 \
	--imgDir=$DATA_ROOT/image \
    --trimapDir=$DATA_ROOT/trimap \
	--alphaDir=$DATA_ROOT/alpha \
	--saveDir=$ROOT/result/tmp \
	--resume=$ROOT/model/debug/ckpt_e200.pth \
	--cuda \
    --stage=1 \
    --arch=vgg16 \
    #--in_chan=3
    #--arch=resnet50_aspp \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_e300 \
        #--not_strict \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_add_shen_e30 \
	#--trimapDir=$DATA_ROOT/test/trimap \
	#--imgDir=$DATA_ROOT/input_lowers \
	#--trimapDir=$DATA_ROOT/trimap_lowers/Trimap1 \
