#! /bin/bash
# https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html
GPU_NUM=2 
CONFIG_FILE=/home/phd/09/igor/mmdetection/configs/ssd/ssd512_sportradar.py
PORT=28003 # port number
EXPERIMENT_DIR=$(pwd)
pwd
echo "Navigating to mmdet..."
cd /home/phd/09/igor/mmdetection/
pwd
MODEL_NAME="checkpoints"
MODEL_DIR=${EXPERIMENT_DIR}/${MODEL_NAME}
echo "${MODEL_NAME} will be saved to"
echo "${MODEL_DIR}"
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} ${PORT} --work_dir ${MODEL_DIR} --gpu-ids 0 1