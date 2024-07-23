#!/bin/bash

SAMPLE_STEP=2000
LOG_STEP=500
if [[ "$2" == "MNIST" ]]; then
	DATASET_NAME="MNIST"
	IMG_SIZE=64
	CLASSES=2
	ITERS=10000
	DATA_DIR="./mnist_im_2_im"
elif [[ "$2" == "micro" ]]; then
	DATASET_NAME="micro"
	IMG_SIZE=64
	CLASSES=2
	ITERS=2
	SAMPLE_STEP=1
	LOG_STEP=1
	DATA_DIR="./horse2zebra64"
elif [[ "$2" == "horse2zebra" ]]; then
	DATASET_NAME="horse2zebra"
	IMG_SIZE=256
	CLASSES=2
	ITERS=200000
	DATA_DIR="./horse2zebra/train"
elif [[ "$2" == "horse2zebra64" ]]; then
	DATASET_NAME="horse2zebra_64x64"
	IMG_SIZE=64
	CLASSES=2
	ITERS=200000
	DATA_DIR="./horse2zebra64"
elif [[ "$2" == "yosemite" ]]; then
	DATASET_NAME="yosemite"
	IMG_SIZE=256
	CLASSES=2
	ITERS=200000
	DATA_DIR="./yosemite"
elif [[ "$2" == "ukiyoe" ]]; then
	DATASET_NAME="ukiyoe"
	IMG_SIZE=128
	CLASSES=2
	ITERS=200000
	DATA_DIR="./ukiyoe"
else
	echo "Invalid dataset."
	exit 1
fi

OUT_DIRECTORY="stargan_${DATASET_NAME}_${1}"

python3 "./stargan-experiments/${1}/main.py" --mode train --dataset RaFD --rafd_crop_size $IMG_SIZE --image_size $IMG_SIZE --c_dim $CLASSES \
        --rafd_image_dir $DATA_DIR --sample_dir "${OUT_DIRECTORY}/samples" --log_dir "${OUT_DIRECTORY}/logs" \
        --model_save_dir "${OUT_DIRECTORY}/models" --result_dir "${OUT_DIRECTORY}/results" \
        --use_tensorboard False \
        --sample_step $SAMPLE_STEP --num_iters $ITERS --model_save_step 50000 --log_step $LOG_STEP