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

import hydra
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import codev
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

def select_reward_fn(data_source):
    return codev.compute_score

def process_responses(args):
    """并行处理单个response的包装函数"""
    reward_fn, prompt, response, ground_truth = args
    return {
        'prompt': prompt,
        'score': reward_fn(response, ground_truth),
    }

@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
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
    with ProcessPoolExecutor(max_workers=256) as executor:
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

        prompt2score_list = defaultdict(list)
        for item in score_lst:
            prompt = item['prompt']
            score = item['score']
            prompt2score_list[prompt].append(score)

        for prompt, score_list in prompt2score_list.items():
            pass_1 += sum(score_list)
            passes += max(score_list)

    print(f'max@{k}: {passes / total}')
    print(f'avg@{k}: {pass_1 / total_1}')

if __name__ == '__main__':
    main()

