#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-deep-image-matting
DATA_ROOT=/home/liuliang/Desktop/alpha_pictures/kisspng

python core/train.py \
	--size_h=320 \
	--size_w=320 \
	--imgDir=$DATA_ROOT/train/image \
	--alphaDir=$DATA_ROOT/train/alpha \
	--fgDir=$DATA_ROOT/train/fg \
	--bgDir=$DATA_ROOT/train/bg \
	--saveDir=$ROOT/model/stage1 \
	--batchSize=4 \
	--nEpochs=20 \
	--step=-1 \
	--lr=0.00001 \
	--wl_weight=0.5 \
	--threads=8 \
	--printFreq=10 \
	--ckptSaveFreq=1 \
	--cuda \
	--resume=model/stage1/ckpt_e5.pth \
        #--pretrain=model/vgg_state_dict.pth
