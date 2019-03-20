#/bin/bash

TEST_DATA_ROOT=/data/datasets/matting/Combined_Dataset/Test_set/comp

python core/deploy.py \
    --size_h=320 \
    --size_w=320 \
    --imgDir=$TEST_DATA_ROOT/image \
    --trimapDir=$TEST_DATA_ROOT/trimap \
    --alphaDir=$TEST_DATA_ROOT/alpha \
    --saveDir=result/stage0 \
    --resume=model/stage0/ckpt_e19.pth \
    --cuda \
    --stage=0 \
    --crop_or_resize=whole \
    --max_size=1600
