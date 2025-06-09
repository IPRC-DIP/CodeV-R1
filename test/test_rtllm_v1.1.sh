#! /bin/bash

# requires vcs
source ~/.bashrc
conda activate codev-r1

# Define your OUT_NAME here
# OUT_NAME=codev_qwen_7b_3.1k_adaptive_dapo_step300-temp_1
CUR_PATH=$(realpath .)
EXTRACT_PATH=$CUR_PATH/results/test/rtllm-v1.1/$OUT_NAME

cd testbench/RTLLM_v1.1
rm -rf ./cache/$OUT_NAME
mkdir -p ./cache/$OUT_NAME
python auto_run_20.py \
    --n 20 \
    --path $EXTRACT_PATH \
    --test_dir ./cache/$OUT_NAME

python change_txt2excel.py \
    --path "./cache/$OUT_NAME/result.txt"
cd ../..
