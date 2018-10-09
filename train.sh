#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-deep-image-matting
#DATA_ROOT=/home/liuliang/Desktop/matting_data/unique_train
DATA_ROOT=/home/liuliang/Desktop/dataset_deep_image_matting/Training_set
TEST_DATA_ROOT=/home/liuliang/Desktop/dataset_deep_image_matting/Test_set

python core/train.py \
	--crop_h=480 \
	--crop_w=480 \
	--size_h=320 \
	--size_w=320 \
	--alphaDir=$DATA_ROOT/alpha  \
	--fgDir=$DATA_ROOT/fg \
	--bgDir=/home/liuliang/Desktop/dataset_voc_bg \
	--saveDir=$ROOT/model/deep_stage1 \
	--batchSize=4 \
	--nEpochs=1000 \
	--step=-1 \
	--lr=0.00001 \
	--wl_weight=0.5 \
	--threads=4 \
	--printFreq=10 \
	--ckptSaveFreq=10 \
	--cuda \
    --stage=1 \
    --arch=vgg16 \
    --pretrain=model/vgg_state_dict.pth \
    --testFreq=10 \
    --testImgDir=$TEST_DATA_ROOT/image \
    --testTrimapDir=$TEST_DATA_ROOT/trimap \
    --testAlphaDir=$TEST_DATA_ROOT/alpha \
    --testResDir=$ROOT/result/tmp \
    #--pretrain=model/deep_img_mat_stage2/ckpt_e200.pth \
        #--arch=resnet50_aspp \
        #--pretrain=model/ckpt-resnet50-19c8e357.pth
        #--resume=model/resnet_aspp_480_stage1/ckpt_e3.pth
        #--in_chan=3  \
