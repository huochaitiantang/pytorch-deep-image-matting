#/bin/bash
DATA_ROOT=/data/datasets/matting/Combined_Dataset
TRAIN_DATA_ROOT=$DATA_ROOT/Training_set/comp
TEST_DATA_ROOT=$DATA_ROOT/Test_set/comp

python core/train.py \
    --crop_h=320,480,640 \
    --crop_w=320,480,640 \
    --size_h=320 \
    --size_w=320 \
    --alphaDir=$TRAIN_DATA_ROOT/alpha  \
    --fgDir=$TRAIN_DATA_ROOT/fg \
    --bgDir=$TRAIN_DATA_ROOT/bg \
    --imgDir=$TRAIN_DATA_ROOT/image \
    --saveDir=model/stage0 \
    --batchSize=1 \
    --nEpochs=25 \
    --step=-1 \
    --lr=0.00001 \
    --wl_weight=0.5 \
    --threads=4 \
    --printFreq=1 \
    --ckptSaveFreq=1 \
    --cuda \
    --stage=0 \
    --pretrain=model/vgg_state_dict.pth \
    --testFreq=1 \
    --testImgDir=$TEST_DATA_ROOT/image \
    --testTrimapDir=$TEST_DATA_ROOT/trimap \
    --testAlphaDir=$TEST_DATA_ROOT/alpha \
    --testResDir=result/tmp \
    --crop_or_resize=whole \
    --max_size=1600
    #--resume=model/stage0/ckpt_e6.pth \
