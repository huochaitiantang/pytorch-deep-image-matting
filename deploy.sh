#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-deep-image-matting
DATA_ROOT=/home/liuliang/Desktop/alpha_pictures/kisspng

python core/deploy.py \
	--size_h=320 \
	--size_w=320 \
	--imgDir=$DATA_ROOT/test/image \
	--trimapDir=$DATA_ROOT/test/trimap \
	--saveDir=$ROOT/result/tmp \
	--alphaDir=$DATA_ROOT/test/alpha \
	--resume=$ROOT/model/stage1/ckpt_e55.pth \
	--cuda

	#--imgDir=/home/liuliang/Desktop/pytorch-fast-matting-portrait/data/images_data_crop
	#--trimapDir=/home/liuliang/Desktop/pytorch-fast-matting-portrait/data/images_trimap_dilated_eroded_size50 
	#--trimapDir=/home/liuliang/Desktop/pytorch-fast-matting-portrait/data/images_deep_image_matting
