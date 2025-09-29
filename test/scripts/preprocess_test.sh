source ~/.bashrc
conda activate codev-r1
mkdir -p data/test
CUR_PATH=$(realpath .)
python -m llmkit_data.cli.prep_verilogeval --version v2 --data_path "$CUR_PATH/testbench/VerilogEval_v2.0.0/spec-to-rtl_0shot_n1.jsonl" --out data/test/verilogeval-v2-spec-to-rtl-codev-rl.jsonl --prompt_type codev_rl
python -m llmkit_data.cli.prep_verilogeval --version v2 --data_path "$CUR_PATH/testbench/VerilogEval_v2.0.0/complete-iccad2023_0shot_n1.jsonl" --out data/test/verilogeval-v2-code-complete-iccad2023-codev-rl.jsonl --prompt_type codev_rl
python -m llmkit_data.cli.prep_verilogeval --version v1 --data_path "$CUR_PATH/testbench/VerilogEval_v1.0.0/data/VerilogEval_Machine.jsonl" --description_path "$CUR_PATH/testbench/VerilogEval_v1.0.0/descriptions/VerilogDescription_Machine.jsonl" --out data/test/verilogeval-v1-machine-codev-rl.jsonl --prompt_type codev_rl
python -m llmkit_data.cli.prep_verilogeval --version v1 --data_path "$CUR_PATH/testbench/VerilogEval_v1.0.0/data/VerilogEval_Human.jsonl" --description_path "$CUR_PATH/testbench/VerilogEval_v1.0.0/descriptions/VerilogDescription_Human.jsonl" --out data/test/verilogeval-v1-human-codev-rl.jsonl --prompt_type codev_rl
python -m llmkit_data.cli.prep_rtllm --version v2 --data_path "$CUR_PATH/testbench/RTLLM_v2.0/rtllm2.jsonl" --out data/test/rtllm-v2-codev-rl.jsonl --prompt_type codev_rl
python -m llmkit_data.cli.prep_rtllm --version v1.1 --data_path "$CUR_PATH/testbench/RTLLM_v1.1/Original_problem" --out data/test/rtllm-v1.1-codev-rl.jsonl --prompt_type codev_rl
