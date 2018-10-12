#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-deep-image-matting
DATA_ROOT=/home/liuliang/Desktop/dataset_deep_image_matting/Training_set
TEST_DATA_ROOT=/home/liuliang/Desktop/dataset_deep_image_matting/Test_set
#DATA_ROOT=/home/liuliang/Desktop/matting_data/unique_train
#TEST_DATA_ROOT=/home/liuliang/Desktop/matting_data/test

python core/train.py \
	--crop_h=320 \
	--crop_w=320 \
	--size_h=320 \
	--size_w=320 \
	--alphaDir=$DATA_ROOT/alpha  \
	--fgDir=$DATA_ROOT/fg \
	--bgDir=/home/liuliang/Desktop/bg_voc+coco_noperson \
	--saveDir=$ROOT/model/debug \
	--batchSize=12 \
	--nEpochs=10000 \
	--step=-1 \
	--lr=0.00001 \
	--wl_weight=0.5 \
	--threads=4 \
	--printFreq=5 \
	--ckptSaveFreq=100 \
	--cuda \
    --stage=1 \
    --arch=vgg16 \
    --pretrain=model/vgg_state_dict.pth \
    --testFreq=100 \
    --testImgDir=$TEST_DATA_ROOT/image \
    --testTrimapDir=$TEST_DATA_ROOT/trimap \
    --testAlphaDir=$TEST_DATA_ROOT/alpha \
    --testResDir=$ROOT/result/tmp \
    #--resume=model/mat_stage1/ckpt_e90.pth \
    #--pretrain=model/deep_img_mat_stage2/ckpt_e200.pth \
    #--arch=resnet50_aspp \
    #--pretrain=model/ckpt-resnet50-19c8e357.pth
    #--in_chan=3  \
