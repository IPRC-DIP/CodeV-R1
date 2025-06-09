import json
import os
import argparse

def process_files(input_file, output_folder):
    countDic = {}
    with open(input_file, 'r', encoding="utf-8") as in_f:
        for line in in_f:
            lineData = json.loads(line.strip())
            task_id = lineData['task_id']
            if task_id in countDic:
                countDic[task_id] += 1
            else:
                countDic[task_id] = 1
            lineData['verilog'] = lineData['completion'].strip()
            # 创建输出文件夹
            task_output_folder = os.path.join(output_folder, "t" + str(countDic[task_id]))
            if not os.path.exists(task_output_folder):
                os.makedirs(task_output_folder)
            # 写入Verilog代码
            with open(os.path.join(task_output_folder, task_id + ".v"), 'w', encoding='utf-8') as wf:
                wf.write(lineData['verilog'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files and generate Verilog files.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder.")
    args = parser.parse_args()

    process_files(args.input_file, args.output_folder)