#/bin/bash
DESKTOP=/home/liuliang/Desktop
ROOT=$DESKTOP/pytorch-deep-image-matting
DATA_ROOT=$DESKTOP/dataset/matting
TRAIN_DATA_ROOT=$DATA_ROOT/dataset_deep_image_matting/Training_set
TEST_DATA_ROOT=$DATA_ROOT/dataset_deep_image_matting/Test_set

python core/train.py \
	--crop_h=320 \
	--crop_w=320 \
	--size_h=320 \
	--size_w=320 \
	--alphaDir=$TRAIN_DATA_ROOT/comp/alpha  \
	--fgDir=$TRAIN_DATA_ROOT/comp/fg \
	--bgDir=$TRAIN_DATA_ROOT/comp/bg \
    --imgDir=$TRAIN_DATA_ROOT/comp/image \
	--saveDir=$ROOT/model/deep_offline_vggnobn \
	--batchSize=1 \
	--nEpochs=10 \
	--step=-1 \
	--lr=0.00001 \
	--wl_weight=0.5 \
	--threads=4 \
	--printFreq=10 \
	--ckptSaveFreq=1 \
	--cuda \
    --stage=1 \
    --arch=vgg16_nobn \
    --dataOffline \
    --pretrain=$ROOT/model/vgg_state_dict.pth \
    --testFreq=1 \
    --testImgDir=$TEST_DATA_ROOT/comp/image \
    --testTrimapDir=$TEST_DATA_ROOT/comp/trimap \
    --testAlphaDir=$TEST_DATA_ROOT/comp/alpha \
    --testResDir=$ROOT/deep_offline_vggnobn \
    #--pretrain=model/input_480_stage3/ckpt_e40.pth \
    #--pretrain=model/deep_img_mat_stage2/ckpt_e200.pth \
    #--arch=resnet50_aspp \
    #--pretrain=model/ckpt-resnet50-19c8e357.pth
    #--in_chan=3  \
