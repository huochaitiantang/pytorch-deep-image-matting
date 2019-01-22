#/bin/bash
DESKTOP=/home/liuliang/Desktop
ROOT=$DESKTOP/pytorch-deep-image-matting-pure
DATA_ROOT=$DESKTOP/dataset/matting
TRAIN_DATA_ROOT=$DATA_ROOT/dataset_deep_image_matting/Training_set
TEST_DATA_ROOT=$DESKTOP/dataset/matting/dataset_deep_image_matting/Test_set

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
    --batchSize=1 \
    --nEpochs=50 \
    --step=-1 \
    --lr=0.00001 \
    --wl_weight=0.5 \
    --threads=4 \
    --printFreq=50 \
    --ckptSaveFreq=1 \
    --cuda \
    --stage=0 \
    --pretrain=model/vgg_state_dict.pth \
    --testFreq=1 \
    --testImgDir=$TEST_DATA_ROOT/comp/image \
    --testTrimapDir=$TEST_DATA_ROOT/comp/trimap \
    --testAlphaDir=$TEST_DATA_ROOT/comp/alpha \
    --testResDir=$ROOT/result/tmp \
    --crop_or_resize=whole \
    --max_size=320 
