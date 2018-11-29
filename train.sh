#/bin/bash
DESKTOP=/data3/liuliang
ROOT=$DESKTOP/pytorch-deep-image-matting
DATA_ROOT=$DESKTOP/data/deep_image_matting
TRAIN_DATA_ROOT=$DATA_ROOT/Train
TEST_DATA_ROOT=$DATA_ROOT/Test

python core/train.py \
    --crop_h=320,480,640 \
    --crop_w=320,480,640 \
    --size_h=320 \
    --size_w=320 \
    --alphaDir=$TRAIN_DATA_ROOT/comp/alpha  \
    --fgDir=$TRAIN_DATA_ROOT/comp/fg \
    --bgDir=$TRAIN_DATA_ROOT/comp/bg \
    --imgDir=$TRAIN_DATA_ROOT/comp/image \
    --saveDir=$ROOT/model/stage0 \
    --batchSize=16 \
    --nEpochs=30 \
    --step=-1 \
    --lr=0.00001 \
    --wl_weight=0.5 \
    --threads=4 \
    --printFreq=1 \
    --ckptSaveFreq=1 \
    --cuda \
    --stage=0 \
    --arch=vgg16_nobn \
    --dataOffline \
    --pretrain=model/vgg_state_dict.pth \
    --testFreq=1 \
    --testImgDir=$TEST_DATA_ROOT/comp/image \
    --testTrimapDir=$TEST_DATA_ROOT/comp/trimap \
    --testAlphaDir=$TEST_DATA_ROOT/comp/alpha \
    --testResDir=$ROOT/result/stage0 \
    --crop_or_resize=crop
