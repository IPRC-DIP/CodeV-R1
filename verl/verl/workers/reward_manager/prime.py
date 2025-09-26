# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from collections import defaultdict


import dill

# 自定义的序列化函数
def wrapper(encoded_func, *args, **kwargs):
    try:
        func = dill.loads(encoded_func)
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error during deserialization: {e}")
        raise


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # 使用 dill 序列化函数
        encoded_func = dill.dumps(evaluation_func)
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(wrapper, encoded_func, task, completion, reference, task_extra_info)
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None


async def parallel_compute_score_async(evaluation_func,
                                       completions,
                                       references,
                                       tasks,
                                       extra_info=None,
                                       num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.)
            for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
        ]
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 reward_fn_key='data_source',
                 max_resp_len=None,
                 overlong_buffer_cfg=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch['prompts']

        response_ids = data.batch['responses']
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = asyncio.run(
                parallel_compute_score_async(self.compute_score,
                                             sequences_str,
                                             ground_truth,
                                             data_sources,
                                             num_processes=64))
        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [0. for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(sequences_str))]
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        data.non_tensor_batch['acc'] = data.batch['acc'].numpy()
        # print('Calculated reward scores in PrimeRewardManager!!!')
        # print('In verify of PrimeRewardManager, keys are', data.batch.keys())
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch['data_source']
        extra_info = data.non_tensor_batch.get('extra_info', [None] * len(data_sources))

        scores = self.verify(data)
        # print('In __call__ of PrimeRewardManager, keys are', data.batch.keys())

        for i in range(len(data)):
            reward = scores[i]
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length[i] - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)


            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
