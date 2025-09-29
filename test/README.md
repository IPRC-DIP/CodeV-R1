## Testing Models on VerilogEval and RTLLM

**TL;DR:** You can run the following scripts to test:

```bash
bash scripts/preprocess_test.sh

MODEL_PATH=YOUR_MODEL_PATH
MODEL_NAME=YOUR_MODEL_NAME

bash benchmark_runner.sh --benchmark rtllm_v1_1 --model_path $MODEL_PATH --model_name $MODEL_NAME
bash benchmark_runner.sh --benchmark rtllm_v2 --model_path $MODEL_PATH --model_name $MODEL_NAME
bash benchmark_runner.sh --benchmark verilogeval_v1 --model_path $MODEL_PATH --model_name $MODEL_NAME
bash benchmark_runner.sh --benchmark verilogeval_v2 --model_path $MODEL_PATH --model_name $MODEL_NAME
```

Below are the details:

Both VerilogEval and RTLLM are located under the `testbench/` directory.
- Preprocessing is uniformly invoked via `scripts/preprocess_test.sh`.
- Sampling + postprocessing are handled in `benchmark_runner.sh`. Tests that don't require `vcs` (VerilogEval) are included in `benchmark_runner.sh`, while those need (RTLLM) have separate shell scripts.

#### VerilogEval Testing (v1 & v2)

- **Preprocessing**: Run `scripts/preprocess_test.sh` (will call `llmkit_data.cli.prep_verilogeval`).
- **Sampling, Postprocessing, and Testing**:
  - VerilogEval v1: refer to `generate_and_test_verilogeval_v1` in `benchmark_runner.sh`.
  - VerilogEval v2: refer to `generate_and_test_verilogeval_v2` in `benchmark_runner.sh`.
  - Both versions use `llmkit_data.cli.sample` for sampling and `scripts/extract_verilog.py` for postprocessing.
- **Results**: 
  - VerilogEval v1: Sampled files and test results are both in `results/test/verilogeval-v1/`.
  - VerilogEval v2: Sampled files are stored in `results/test/verilogeval-v2/`, while test results are in `testbench/VerilogEval_v2.0.0/results/`.


#### RTLLM Testing

- **Preprocessing**: Run `scripts/preprocess_test.sh`, invoking `llmkit_data.cli.prep_rtllm`.
- **Local sampling + Postprocessing**: Refer to `generate_rtllm_v1.1` or  `generate_rtllm_v2` in `benchmark_runner.sh`, calling `llmkit_data.cli.sample` + `scripts/extract_verilog.py`.
- **Testing**: Run `test_rtllm_v1.1.sh` or `test_rtllm_v2.sh`. They require `vcs`, which might be on other nodes. Set `OUT_NAME` to the corresponding jsonl file under `results/test/rtllm-v2`.
- **Results**:
  - RTLLM v1.1: Sampled files are under `results/test/rtllm_v1_1/`, while test results are under `testbench/RTLLM_v1.1/cache/`.
  - RTLLM v2: Sampled files are under `results/test/rtllm_v2/`, while test results are under `results/test/rtllm_v2/`.
