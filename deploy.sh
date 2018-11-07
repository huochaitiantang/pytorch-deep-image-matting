#/bin/bash

DESKTOP=/home/liuliang/Desktop

MODEL_ROOT=$DESKTOP/pytorch-deep-image-matting
#DATA_ROOT=/home/liuliang/Desktop/matting_data/test
#DATA_ROOT=/home/liuliang/Desktop/dataset_shen_matting
#DATA_ROOT=/home/liuliang/Desktop/dataset_alpha_matting
#DATA_ROOT=/home/liuliang/Desktop/dataset_deep_image_matting/Test_set
#DATA_ROOT=/home/liuliang/Desktop/photo_test
DATA_ROOT=$DESKTOP/dataset/matting/matting_data/comp_with_our_photo_test

python core/deploy.py \
	--size_h=512 \
	--size_w=384 \
	--imgDir=$DATA_ROOT/image \
    --trimapDir=$DATA_ROOT/trimap \
	--saveDir=$MODEL_ROOT/result/comp_with_our_photo \
    --alphaDir=$DATA_ROOT/alpha \
	--resume=$MODEL_ROOT/model/comp_with_our_photo/ckpt_e1.pth \
	--cuda \
    --stage=1 \
    --arch=vgg16_nobn \
    #--in_chan=3
    #--arch=resnet50_aspp \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_e300 \
        #--not_strict \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_add_shen_e30 \
	#--trimapDir=$DATA_ROOT/test/trimap \
	#--imgDir=$DATA_ROOT/input_lowers \
	#--trimapDir=$DATA_ROOT/trimap_lowers/Trimap1 \
	#--alphaDir=$DATA_ROOT/alpha \
