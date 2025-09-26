# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import json
import hydra
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import codev
from verl.utils.reward_score.codev_eval_toolkit.eval_codev import extract_verilog
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict
import openai


import os
from datasets import Dataset, load_dataset

with open("/nfs_global/S/zhangxiaoyun/dapo/scripts/testbench_gen/generate.prompt", "r") as f:
    template = f.read()

def ask_gpt_question(prompt):
    # openai.api_key = 'sk-hUdHhJARgRIzyAtOITzrehIYEUdtqiBEXAAQzdmCV2JRsZpu'
    # openai.base_url = 'https://api.lkeap.cloud.tencent.com/v1/'
    openai.api_key = 'eddb32b2-7673-476d-bb7d-b3013cc14a3d'
    openai.base_url = 'https://ark.cn-beijing.volces.com/api/v3/'
    prompt = template.format_map({"our_code":prompt})

    try:
        response = openai.chat.completions.create(
            model='deepseek-v3-241226',
            # model='gpt-4o',
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return str(e)
    
def extract_problem_section(text):
    if "[Solution]" in text:
        start_index = text.find("[Solution]")
        if start_index != -1:
            text = text[:start_index]
        return extract_problem_section(text)
    
    if "### Solution" in text:
        start_index = text.find("### Solution")
        if start_index != -1:
            text = text[:start_index]
        return extract_problem_section(text)
    
    if "[Problem]" in text:
        parts = text.split('[Problem]')
        if len(parts) < 2:
            return None
        problem_section = parts[1].strip()
        return problem_section

    if "### Problem" in text:
        parts = text.split('### Problem')
        if len(parts) < 2:
            return None
        problem_section = parts[1].strip()
        return problem_section

    return None

def process_entry(code):
    """
    处理单个条目：
    1. 调用ask_gpt_question生成response
    2. 将结果写入输出文件（线程安全）
    """
    if code is None:
        return  # 跳过无效条目
    
    try:
        response_ = ask_gpt_question(code)
        response = extract_problem_section(response_)
        while response is None:
            response_ = ask_gpt_question(code)
            response = extract_problem_section(response_)

        return {
            "instruction": response,
            "response": code
        }
    except Exception as e:
        print(f"Error processing code {code}: {e}")

def mk_prompt_r1(question):
    question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")
    system_prompt = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to write verilog code. After thinking, when you finally reach a conclusion, enclose the final verilog code in ```verilog ``` within <answer> </answer> tags. i.e., <answer> ```verilog\n module top_module(in, out, ...) ... ``` </answer>.\n"""
    user_prompt = question.strip() + "\n"
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return conversation

def select_reward_fn(data_source):
    return codev.compute_score

def process_responses(args):
    """并行处理单个response的包装函数"""
    reward_fn, prompt, response, ground_truth = args
    return {
        'prompt': prompt,
        'response': response,
        'score': reward_fn(response, ground_truth),
    }

def majority_vote(code_list, equal_judge_func):
    # 多数投票函数
    # 利用等价判断函数判断两个代码片段是否相等，然后根据多数投票返回最多的结果
    count = {}
    for code in code_list:
        for other_code in code_list:
            if equal_judge_func(code, other_code):
                count[code] = count.get(code, 0) + 1
            else:
                count[code] = count.get(code, 0)

    # 返回投票最多的code片段
    return max(count.items(), key=lambda x: x[1])[0]


def extract_code_from_response(response):
    # 提取代码片段的函数
    return extract_verilog(response)

def process_responses_for_vote(args):
    responses = args
    majority_vote_code_response = majority_vote(responses, codev.compute_score)
    majority_vote_code = extract_code_from_response(majority_vote_code_response)
    return majority_vote_code
    

@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    output_path = config.data.output_path
    dataset = pd.read_parquet(local_path)
    print(len(dataset))
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    k = len(responses[0])
    pass_1 = 0
    total_1 = total * k

    # 创建进程池（根据CPU核心数调整max_workers）
    with ProcessPoolExecutor(max_workers=64) as executor:
        print("Processing responses...")
        task_args = []
        for i in range(total):
            prompt = prompts[i]
            prompt = str(prompt.tolist())

            response_lst = responses[i]
            data_source = data_sources[i]
            reward_fn = select_reward_fn(data_source)
            ground_truth = reward_model_data[i]['ground_truth']

            # 准备并行任务参数
            task_args.extend([(reward_fn, prompt, r, ground_truth) for r in response_lst])
        # 并行执行计算
        score_lst = list(executor.map(process_responses, task_args))

    # 开始计算结果
    prompt2score_list = defaultdict(list)
    prompt2response_list = defaultdict(list)
    for item in score_lst:
        prompt = item['prompt']
        response = item['response']
        score = item['score']
        prompt2score_list[prompt].append(score)
        prompt2response_list[prompt].append(response)

    failed_response_list = []
    for prompt, score_list in prompt2score_list.items():
        pass_1 += sum(score_list)
        passes += max(score_list)

        if max(score_list) == 0:
            failed_response_list.append(prompt2response_list[prompt])
    
    print(f'max@{k}: {passes / total}')
    print(f'avg@{k}: {pass_1 / total_1}')
    print(f"failed response list: {len(failed_response_list)}")
    
    failed_code_list = []
    with ProcessPoolExecutor(max_workers=64) as executor:
        print("Processing failed code list to majortiy vote...")
        task_args = []
        task_args.extend([(rs) for rs in failed_response_list])
        failed_code_list = list(executor.map(process_responses_for_vote, task_args))

    new_dataset = []
    with ThreadPoolExecutor(max_workers=2048) as executor:
        # 并行执行计算
        print("Processing failed code list to ask gpt...")
        new_dataset = list(executor.map(process_entry, failed_code_list))
    
    # new_dataset是一个列表，每个元素都是一个字典，包含了数据集中的一个条目
    # 将new_dataset转换为transformers库的dataset
    print("Processing failed code list to new dataset...")
    new_dataset = Dataset.from_list(new_dataset)
    
    def process_fn(example, idx):
        question = example["instruction"]
        ground_truth = example["response"]
        
        data = {
            "data_source": "codev",
            "prompt": mk_prompt_r1(question),
            "ability": "verilog",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
            }
        }
        return data
    new_dataset = new_dataset.map(function=process_fn, with_indices=True)

    print(new_dataset[0])

    # 需不需要对new_dataset再做一次处理，保证code可综合？
    new_dataset.to_parquet(output_path)

if __name__ == '__main__':
    main()

