import os
import glob
from pathlib import Path
import json
import argparse
from llmkit_data.utils.json import read_jsonl, write_jsonl
from llmkit_data.cli.prep_verilogeval import mk_prompt_o1_training, mk_prompt_o1_general, mk_prompt_r1, mk_prompt_r1_user, mk_prompt_codev_rl


def convert_to_sft_v1(data_path, prompt_type):
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path) and not item.startswith("_"):
            # 构建要提取的文件路径
            description_file = os.path.join(item_path, 'design_description.txt')
            # verilog_file = os.path.join(item_path, f'verified_{item}.v')
            verilog_file = glob.glob(os.path.join(item_path, 'verified_*.v'))[0]

            # 提取 design_description.txt 的内容
            question = ""
            if os.path.exists(description_file):
                try:
                    with open(description_file, 'r', encoding='utf-8') as f:
                        question = f.read()
                except Exception as e:
                    print(f"读取 {description_file} 时出错: {e}")

            # 提取 verified_{文件夹名}.v 的内容
            golden_code = ""
            if os.path.exists(verilog_file):
                try:
                    with open(verilog_file, 'r', encoding='utf-8') as f:
                        golden_code = f.read()
                except Exception as e:
                    print(f"读取 {verilog_file} 时出错: {e}")

            if prompt_type == "simple":
                question = question
            elif prompt_type == "o1_general":
                question = mk_prompt_o1_general(question)
            elif prompt_type == "o1_training":
                question = mk_prompt_o1_training(question)
            elif prompt_type == "r1":
                question = mk_prompt_r1(question)
            elif prompt_type == "codev_rl":
                question = mk_prompt_codev_rl(question)
            else:
                raise NotImplementedError(f"Prompt type {prompt_type} not supported!")

            yield {"task_id": item, "question": question, "golden_code": golden_code}


def convert_to_sft_v2(data_path, prompt_type):
    for task in read_jsonl(data_path):
        question = task["prompt"]
        if prompt_type == "simple":
            question = question
        elif prompt_type == "o1_general":
            question = mk_prompt_o1_general(question)
        elif prompt_type == "o1_training":
            question = mk_prompt_o1_training(question)
        elif prompt_type == "r1":
            question = mk_prompt_r1(question)
        elif prompt_type == "codev_rl":
            question = mk_prompt_codev_rl(question)
        else:
            raise NotImplementedError(f"Prompt type {prompt_type} not supported!")

        yield {"question": question, "task_id": task["task_id"]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v2", help="verilog-eval version", choices=["v1.1", "v2"])
    parser.add_argument("--data_path", type=str, help="rtllm data path")
    parser.add_argument("--out", type=str, help="output path")
    parser.add_argument("--prompt_type", type=str, choices=["simple", "o1_general", "o1_training", "r1", "codev_rl"], help="One of `simple`, `o1_general`, `o1_training`, `r1`, `codev_rl`")
    args = parser.parse_args()

    if args.version == "v1.1":
        write_jsonl(convert_to_sft_v1(args.data_path, args.prompt_type), args.out)
    else:
        write_jsonl(convert_to_sft_v2(args.data_path, args.prompt_type), args.out)