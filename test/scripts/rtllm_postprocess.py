from collections import defaultdict
import re
import os
import argparse
import json


def split_file(source_path, output_dir):
    task_id_counter = defaultdict(int)
    with open(source_path, "r") as f:
        samples = list(map(json.loads, f.read().strip().splitlines()))
    
    for sample in samples:
        # len(codes)==0: response too long, not ended
        # len(codes)>=2: have code when thinking
        # if len(codes) > 1:
        #     print("!!!!!!!!!!!!")
        #     print(len(codes))
        #     print(sample['response'][0]['content'])

        task_id = sample["task_id"]
        code = sample["completion"]
        task_id_counter[task_id] += 1
        out_file = f"{output_dir}/t{task_id_counter[task_id]}/{task_id}.v"
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        with open(out_file, "w") as f:
            f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="results/test/rtllm-v2/codev_qwen_7b_rl_618271-temp_0.6.jsonl", help="source result jsonl")
    parser.add_argument("--out", type=str, default="tmp/tmp", help="output path")
    args = parser.parse_args()
    split_file(args.source, args.out)
