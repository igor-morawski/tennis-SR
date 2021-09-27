#! /bin/bash
GPU_NUM=1 # ${GPU_NUM}
CONFIG_FILE=/home/phd/09/igor/mmdetection/configs/ssd/ssd512_sportradar.py
# ${CONFIG_FILE}
EXPERIMENT_DIR=$(pwd)
pwd
echo "Navigating to mmdet..."
cd /home/phd/09/igor/mmdetection/
pwd
CHECKPOINT_FILE=/tmp2/igor/tennis-SR/checkpoints/epoch_9.pth
echo "Testing ${CONFIG_FILE}, checkpoint ${CHECKPOINT_FILE}"
bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} 33542 --eval bbox 