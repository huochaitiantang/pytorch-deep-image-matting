#/bin/bash

DESKTOP=/home/liuliang/Desktop
MODEL_ROOT=$DESKTOP/pytorch-deep-image-matting-pure
DATA_ROOT=$DESKTOP/dataset/matting/dataset_deep_image_matting/Test_set

python core/deploy.py \
    --size_h=320 \
    --size_w=320 \
    --imgDir=$DATA_ROOT/comp/image \
    --trimapDir=$DATA_ROOT/comp/trimap \
    --alphaDir=$DATA_ROOT/comp/alpha \
    --saveDir=$MODEL_ROOT/result/tmp \
    --resume=$MODEL_ROOT/model/batch1_stage0_ckpt_e22.pth \
    --cuda \
    --stage=0 \
    --crop_or_resize=whole \
    --max_size=320
