#/bin/bash

TEST_DATA_ROOT=/home/liuliang/DISK_2T/datasets/matting/Combined_Dataset/Test_set/comp

CUDA_VISIBLE_DEVICES=0 \
python core/deploy.py \
    --imgDir=$TEST_DATA_ROOT/image \
    --trimapDir=$TEST_DATA_ROOT/trimap \
    --alphaDir=$TEST_DATA_ROOT/alpha \
    --saveDir=result/tmp \
    --resume=model/stage1_sad_57.1.pth \
    --cuda \
    --stage=1 \
    --crop_or_resize=whole \
    --max_size=1600
