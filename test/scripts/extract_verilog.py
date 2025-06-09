import re
import json
import argparse
from utils.verilog_extractor import extract_verilog_code


def extract_verilog_from_file(sample_path, extracted_path, remove_head=False, add_backtick=False):
    # sample_path = f"results/test/{name}.jsonl"
    # extracted_path = f"results/test/{name}-extracted.jsonl"
    with open(sample_path, 'r') as f:
        data = list(map(json.loads, f.read().strip().splitlines()))
        data.sort(key=lambda x: x["task_id"])
    # 别在这里写入，跨账户的时候可能会出权限问题
    # with open(sample_path, 'w') as f:
    #     f.write('\n'.join(map(json.dumps, data)) + '\n')

    for i in range(len(data)):
        # data[i]["completion"] = extract_verilog_code(data[i]['completion'])
        if "completion" in data[i]:
            completion = data[i]["completion"]
        elif "response" in data[i] and "content" in data[i]["response"][0]:
            completion = data[i]["response"][0]["content"]
        data[i] = {"task_id": data[i]["task_id"],
                   "completion": extract_verilog_code(completion, remove_head, add_backtick)}

    with open(extracted_path, 'w') as f:
        f.write('\n'.join(map(json.dumps, data)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of store_true")
    parser.add_argument('--in_path', type=str, help="Path of samples")
    parser.add_argument('--out_path', type=str, help="Path of extracted code")
    parser.add_argument('--remove_head', action="store_true", help="Remove module head")
    parser.add_argument('--add_backtick', action="store_true", help="Add back backticks to deal with verilog-eval extraction")
    args = parser.parse_args()

    # prefix: "verilogeval-v2"; model: ["dsv3", "dsv3-temp1", "qwq-temp0.2", "qwq-temp0.6", "dsr1-qwen-32b", "dsr1", "qwencoder-32b"]
    # extract_verilog_from_file("verilogeval-v2-codev-qwen32b-o1")
    extract_verilog_from_file(args.in_path, args.out_path, args.remove_head, args.add_backtick)