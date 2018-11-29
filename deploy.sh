#/bin/bash

DESKTOP=/data3/liuliang

MODEL_ROOT=$DESKTOP/pytorch-deep-image-matting
DATA_ROOT=$DESKTOP/data/deep_image_matting/Test


python core/deploy.py \
    --size_h=320 \
    --size_w=320 \
    --imgDir=$DATA_ROOT/comp/image \
    --trimapDir=$DATA_ROOT/comp/trimap \
    --alphaDir=$DATA_ROOT/comp/alpha \
    --saveDir=$MODEL_ROOT/result/tmp \
    --resume=$MODEL_ROOT/model/stage0/ckpt_e1.pth \
    --cuda \
    --stage=0 \
    --arch=vgg16_nobn \
    --crop_or_resize=crop
