#!/bin/bash

set -x
ulimit -l unlimited
ulimit -v unlimited
source ~/.bashrc
conda activate codev-r1
export VLLM_USE_V1=1
export CUDA_LAUNCH_BLOCKING=1

generate_and_test_verilogeval_v1_subset(){
    MODEL_PATH=$1
    MODEL=$2
    TYPE=$3
    TEMPERATURE=$4
    
    DATA_PATH=data/test/verilogeval-v1-$TYPE-codev-rl.jsonl
    SHOTS=0
    SAMPLES=20
    # TOP_P=0.95
    TOP_P=1

    OUT_NAME=$MODEL-shot_$SHOTS-temp_$TEMPERATURE.jsonl
    SAMPLE_PATH=$(realpath .)/results/test/verilogeval-v1/$TYPE/raw/$OUT_NAME
    EXTRACT_PATH=$(realpath .)/results/test/verilogeval-v1/$TYPE/$OUT_NAME

    echo "Start to sample. Output path is $SAMPLE_PATH"
    python -m llmkit_data.cli.sample --prompts $DATA_PATH --out $SAMPLE_PATH --model $MODEL_PATH --n_sample $SAMPLES --max_tokens 16384 --gpu_per_model 4 --temperature $TEMPERATURE #--enforce_eager
    echo "Extract verilog code from samples. Output path is $EXTRACT_PATH"
    python scripts/extract_verilog.py --in_path $SAMPLE_PATH --out_path $EXTRACT_PATH --remove_head

    cd testbench/VerilogEval_v1.0.0
    evaluate_functional_correctness $EXTRACT_PATH --problem_file=data/VerilogEval_${TYPE^}.jsonl
    cd ../..
}
generate_and_test_verilogeval_v1() {
    MODEL_PATH=$1
    MODEL=$2
    generate_and_test_verilogeval_v1_subset $MODEL_PATH $MODEL machine 1.0
    generate_and_test_verilogeval_v1_subset $MODEL_PATH $MODEL human 1.0
}

generate_and_test_verilogeval_v2_task() {
    MODEL_PATH=$1
    MODEL=$2
    TASK=$3
    TEMPERATURE=$4

    DATA_PATH=data/test/verilogeval-v2-$TASK-codev-rl.jsonl
    SHOTS=0
    SAMPLES=20
    # TOP_P=0.95
    TOP_P=1

    TASK_=$(echo $TASK | sed 's/-/_/g')
    OUT_NAME=$MODEL-$TASK_-shot_$SHOTS-temp_$TEMPERATURE.jsonl
    SAMPLE_PATH=results/test/verilogeval-v2/raw/$OUT_NAME
    EXTRACT_PATH=results/test/verilogeval-v2/$OUT_NAME

    echo "Start to sample. Output path is $SAMPLE_PATH"
    python -m llmkit_data.cli.sample --prompts $DATA_PATH --out $SAMPLE_PATH --model $MODEL_PATH --n_sample $SAMPLES --max_tokens 16384 --gpu_per_model 4 --temperature $TEMPERATURE #--enforce_eager
    echo "Extract verilog code from samples. Output path is $EXTRACT_PATH"

    if [ "$TASK" == "spec-to-rtl" ]; then
        python scripts/extract_verilog.py --in_path $SAMPLE_PATH --out_path $EXTRACT_PATH --add_backtick
    else
        python scripts/extract_verilog.py --in_path $SAMPLE_PATH --out_path $EXTRACT_PATH --remove_head --add_backtick
    fi

    rm -r testbench/VerilogEval_v2.0.0/results/$MODEL-$TASK-shot_$SHOTS-sample_$SAMPLES-temp_$TEMPERATURE
    
    mkdir -p testbench/VerilogEval_v2.0.0/results
    path=$(realpath "testbench/VerilogEval_v2.0.0/results/$MODEL-$TASK-shot_$SHOTS-sample_$SAMPLES-temp_$TEMPERATURE")
    echo "Begin to test verilog-eval v2. The path containing test files is $path."
    mkdir -p $path
    cd $path

    ../../configure  --with-task=$TASK --with-model=$MODEL --with-examples=$SHOTS --with-samples=$SAMPLES --with-temperature=$TEMPERATURE --with-top-p=$TOP_P
    make -j16
    cd ../..
    python calc_pass_k.py --result_file "$path/summary.csv"
    cd ../..
}
generate_and_test_verilogeval_v2() {
    MODEL_PATH=$1
    MODEL=$2
    generate_and_test_verilogeval_v2_task $MODEL_PATH $MODEL spec-to-rtl 1.0
    generate_and_test_verilogeval_v2_task $MODEL_PATH $MODEL code-complete-iccad2023 1.0
}

generate_rtllm_v1_1() {
    MODEL_PATH=$1
    MODEL=$2
    CUR_PATH=$(realpath .)

    DATA_PATH=data/test/rtllm-v1.1-codev-rl.jsonl
    SAMPLES=20
    # TEMPERATURE=0.6
    TEMPERATURE=1
    TOP_P=0.95
    # TOP_P=1.0

    OUT_NAME=$MODEL-temp_$TEMPERATURE
    SAMPLE_PATH=$CUR_PATH/results/test/rtllm-v1.1/raw/$OUT_NAME.jsonl
    EXTRACT_1_PATH=$CUR_PATH/results/test/rtllm-v1.1/$OUT_NAME.jsonl
    python -m llmkit_data.cli.sample --prompts $DATA_PATH --out $SAMPLE_PATH --model $MODEL_PATH --n_sample $SAMPLES --max_tokens 16384 --gpu_per_model 4 --temperature $TEMPERATURE
    python scripts/extract_verilog.py --in_path $SAMPLE_PATH --out_path $EXTRACT_1_PATH

    EXTRACT_PATH=$CUR_PATH/results/test/rtllm-v1.1/$OUT_NAME
    python scripts/rtllm_postprocess.py --source $EXTRACT_1_PATH --out $EXTRACT_PATH
}

generate_rtllm_v2() {
    MODEL_PATH=$1
    MODEL=$2
    CUR_PATH=$(realpath .)

    DATA_PATH=data/test/rtllm-v2-codev-rl.jsonl
    SAMPLES=20
    # TEMPERATURE=0.6
    TEMPERATURE=1
    TOP_P=0.95
    # TOP_P=1.0

    OUT_NAME=$MODEL-temp_$TEMPERATURE
    SAMPLE_PATH=$CUR_PATH/results/test/rtllm-v2/raw/$OUT_NAME.jsonl
    EXTRACT_1_PATH=$CUR_PATH/results/test/rtllm-v2/$OUT_NAME.jsonl
    python -m llmkit_data.cli.sample --prompts $DATA_PATH --out $SAMPLE_PATH --model $MODEL_PATH --n_sample $SAMPLES --max_tokens 16384 --gpu_per_model 4 --temperature $TEMPERATURE
    python scripts/extract_verilog.py --in_path $SAMPLE_PATH --out_path $EXTRACT_1_PATH

    EXTRACT_PATH=$CUR_PATH/results/test/rtllm-v2/$OUT_NAME
    python scripts/rtllm_postprocess.py --source $EXTRACT_1_PATH --out $EXTRACT_PATH
}


# Function to display usage information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --benchmark <name>    Specify benchmark type (verilogeval_v1, verilogeval_v2, rtllm_v1_1, rtllm_v2)"
    echo "  --model_path <path>   Specify model path"
    echo "  --model_name <name>   Specify model name"
    echo "  --help                Show this help message"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Check required arguments
if [ -z "$BENCHMARK" ] || [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Error: --benchmark, --model_path, and --model_name are required"
    show_help
    exit 1
fi

# Execute corresponding benchmark
case $BENCHMARK in
    verilogeval_v1)
        generate_and_test_verilogeval_v1 "$MODEL_PATH" "$MODEL_NAME"
        ;;
    verilogeval_v2)
        generate_and_test_verilogeval_v2 "$MODEL_PATH" "$MODEL_NAME"
        ;;
    rtllm_v1_1)
        generate_rtllm_v1_1 "$MODEL_PATH" "$MODEL_NAME"
        ;;
    rtllm_v2)
        generate_rtllm_v2 "$MODEL_PATH" "$MODEL_NAME"
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        show_help
        exit 1
        ;;
esac