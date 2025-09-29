#! /bin/bash

# requires vcs
source ~/.bashrc
conda activate codev-r1

run_task() {
  local OUT_NAME=$1
  local CUR_PATH=$(realpath .)
  local EXTRACT_PATH="$CUR_PATH/results/test/rtllm-v2/$OUT_NAME"
  local JSONLPATH="$CUR_PATH/results/test/rtllm-v2/result/$OUT_NAME.jsonl"
  local TXTPATH="$CUR_PATH/results/test/rtllm-v2/result/$OUT_NAME.txt"
  
  rm -f $JSONLPATH $TXTPATH
  rm -rf testbench/RTLLM_v2.0_$OUT_NAME
  cp -r testbench/RTLLM_v2.0 testbench/RTLLM_v2.0_$OUT_NAME
  
  cd testbench/RTLLM_v2.0_$OUT_NAME
  python auto_run.py --path $EXTRACT_PATH --jsonloutpath $JSONLPATH &> $TXTPATH
  cd ../..
  
  rm -rf testbench/RTLLM_v2.0_$OUT_NAME
}

# Define your OUT_NAME array here
# OUT_NAMES=(
#   "deepseek_r1_0528_qwen3_8b-temp_0.6"
# )

for OUT_NAME in "${OUT_NAMES[@]}"; do
  echo "Running task for $OUT_NAME"
  run_task "$OUT_NAME"
done