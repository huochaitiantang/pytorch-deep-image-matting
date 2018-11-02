#/bin/bash

DESKTOP=/home/liuliang/Desktop

MODEL_ROOT=$DESKTOP/pytorch-deep-image-matting
#DATA_ROOT=/home/liuliang/Desktop/matting_data/test
#DATA_ROOT=/home/liuliang/Desktop/dataset_shen_matting
#DATA_ROOT=/home/liuliang/Desktop/dataset_alpha_matting
#DATA_ROOT=/home/liuliang/Desktop/dataset_deep_image_matting/Test_set
#DATA_ROOT=/home/liuliang/Desktop/photo_test
DATA_ROOT=$DESKTOP/dataset/matting/xuexin

python core/deploy.py \
	--size_h=512 \
	--size_w=384 \
	--imgDir=$DATA_ROOT/complex \
    --trimapDir=$DATA_ROOT/complex_trimap_fintune_all_data \
	--saveDir=$MODEL_ROOT/result/tmp \
	--resume=$MODEL_ROOT/model/fintune_with_all_data/ckpt_e100.pth \
	--cuda \
    --stage=3 \
    --arch=vgg16 \
    #--in_chan=3
    #--arch=resnet50_aspp \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_e300 \
        #--not_strict \
	#--trimapDir=/home/liuliang/Desktop/pytorch-alpha-matting/result/trimap_vgg_add_shen_e30 \
	#--trimapDir=$DATA_ROOT/test/trimap \
	#--imgDir=$DATA_ROOT/input_lowers \
	#--trimapDir=$DATA_ROOT/trimap_lowers/Trimap1 \
	#--alphaDir=$DATA_ROOT/alpha \
