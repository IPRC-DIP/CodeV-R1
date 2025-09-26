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
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List, Callable
import numpy as np
from verl import DataProto
from collections import Counter, defaultdict
from transformers import AutoTokenizer


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


import pandas as pd
import json
import multiprocessing
lock = multiprocessing.Lock()
def compute_reward_metrics(batch):
    data_source = batch.non_tensor_batch['data_source']
    index = batch.non_tensor_batch['uid']
    reward_tensor = batch.batch['token_level_scores'].sum(-1)
    reward_metrics = {}
    zero_correct_indices = []

    # TODO: consider dataset mixing
    if data_source[0] == 'codev':
        reward_correct = 1.0
        reward_wrong_answer = 0.0
        reward_format_error = 0.0
    else:
        return reward_metrics
        
    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
    all_correct = torch.sum(reward_tensor == reward_correct).float() / reward_tensor.numel()
    reward_metrics["reward/all_correct_ratio"] = all_correct.detach().item()
    wrong_answer = torch.sum(reward_tensor == reward_wrong_answer).float() / reward_tensor.numel()
    reward_metrics["reward/wrong_answer_ratio"] = wrong_answer.detach().item()
    if reward_format_error is not None and reward_format_error != reward_wrong_answer:
        format_error = torch.sum(reward_tensor == reward_format_error).float() / reward_tensor.numel()
        reward_metrics["reward/format_error_ratio"] = format_error.detach().item()

    correct = (reward_tensor == reward_correct).float()
    correct_series = pd.Series(correct.numpy())

    df = pd.DataFrame({
        'index': index,
        'correct': correct_series
    })

    average_dict = df.groupby('index')['correct'].mean().to_dict()

    for idx, rate in average_dict.items():
        if rate == 0:
            zero_correct_indices.append(idx)

    correct_rates = list(average_dict.values())
    correct_rates_series = pd.Series(correct_rates)

    bins = [0, 0.00001, 0.5, 0.99999, 1.00001]
    labels = ['0%', '(0%, 50%)', '[50%, 100%)', '100%']

    categories = pd.cut(correct_rates_series, bins=bins, labels=labels, right=False)

    proportions = categories.value_counts(normalize=True).to_dict()
    reward_metrics["reward/correct_0%_ratio"] = proportions['0%']
    reward_metrics["reward/correct_(0%,50%)_ratio"] = proportions['(0%, 50%)']
    reward_metrics["reward/correct_[50%,100%)_ratio"] = proportions['[50%, 100%)']
    reward_metrics["reward/correct_100%_ratio"] = proportions['100%']

    print('Correct proportions:', proportions)
    print('Zero correct indices:', zero_correct_indices)

    return reward_metrics

def compute_reflection_metrics(batch: DataProto, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    reflection_words = [
        "verify", "check", "confirm", "however", "reflect", "wait", 
        "correct", "revise", "adjust", "re-evaluate", "re-examine", "yet"
    ]
    
    response_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch['responses']]

    response_info = _compute_response_info(batch)
    response_length = response_info['response_length']

    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    is_codev = batch.non_tensor_batch.get('data_source', [''])[0] == 'codev'
    
    if is_codev:
        reward_correct = 1.0
        correct = (reward_tensor == reward_correct).float()
    else:
        correct = (reward_tensor > 0).float()
    
    metrics = {}
    
    contains_any_reflection = np.zeros(len(response_text), dtype=bool)
    word_presence = {word: np.zeros(len(response_text), dtype=bool) for word in reflection_words}
    word_counts = {word: 0 for word in reflection_words}
    total_words = 0
    
    for i, text in enumerate(response_text):
        text_lower = text.lower()
        for word in reflection_words:
            count = text_lower.count(word)
            if count > 0:
                word_presence[word][i] = True
                contains_any_reflection[i] = True
                word_counts[word] += count
                total_words += count
    
    metrics["reflection/any_word_frequency"] = total_words

    if np.any(contains_any_reflection):
        metrics["reflection/with_length_mean"] = float(np.mean(response_length[contains_any_reflection].cpu().numpy()))
        metrics["reflection/without_length_mean"] = float(np.mean(response_length[~contains_any_reflection].cpu().numpy()))
    
    if np.any(contains_any_reflection):
        metrics["reflection/with_correct_ratio"] = float(np.mean(correct[contains_any_reflection].cpu().numpy()))
        metrics["reflection/without_correct_ratio"] = float(np.mean(correct[~contains_any_reflection].cpu().numpy()))
    
    if np.any(contains_any_reflection):
        metrics["reflection/with_reward_mean"] = float(np.mean(reward_tensor[contains_any_reflection].cpu().numpy()))
        metrics["reflection/without_reward_mean"] = float(np.mean(reward_tensor[~contains_any_reflection].cpu().numpy()))
    
    for word in reflection_words:
        metrics[f"reflection_{word}/word_{word}_frequency"] = word_counts[word]
        metrics[f"reflection_{word}/with_{word}_length_mean"] = float(np.mean(response_length[word_presence[word]].cpu().numpy()))
        metrics[f"reflection_{word}/without_{word}_length_mean"] = float(np.mean(response_length[~word_presence[word]].cpu().numpy()))
        metrics[f"reflection_{word}/with_{word}_correct_ratio"] = float(np.mean(correct[word_presence[word]].cpu().numpy()))
        metrics[f"reflection_{word}/without_{word}_correct_ratio"] = float(np.mean(correct[~word_presence[word]].cpu().numpy()))
        metrics[f"reflection_{word}/with_{word}_reward_mean"] = float(np.mean(reward_tensor[word_presence[word]].cpu().numpy()))
        metrics[f"reflection_{word}/without_{word}_reward_mean"] = float(np.mean(reward_tensor[~word_presence[word]].cpu().numpy()))
    
    return metrics

def compute_language_mix_metrics(batch: DataProto, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    response_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch['responses']]
    reward_tensor = batch.batch['token_level_scores'].sum(-1)
    response_info = _compute_response_info(batch)
    response_length = response_info['response_length']
    
    is_codev = 'is_codev' in batch.batch and batch.batch['is_codev']
    
    if is_codev:
        reward_correct = 1.0
        correct = (reward_tensor == reward_correct).float()
    else:
        correct = (reward_tensor > 0).float()
    
    metrics = {}
    
    contains_mix = np.zeros(len(response_text), dtype=bool)
    total_mix_count = 0
    
    for i, text in enumerate(response_text):
        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_count = sum(1 for char in text if ('a' <= char.lower() <= 'z'))
        
        if chinese_count > 0 and english_count > 0:
            contains_mix[i] = True
            total_mix_count += 1
    
    metrics["language_mix/frequency"] = total_mix_count
    metrics["language_mix/ratio"] = float(np.mean(contains_mix))
    
    if np.any(contains_mix):
        metrics["language_mix/with_length_mean"] = float(np.mean(response_length[contains_mix].cpu().numpy()))
        metrics["language_mix/without_length_mean"] = float(np.mean(response_length[~contains_mix].cpu().numpy()))
    
        metrics["language_mix/with_correct_ratio"] = float(np.mean(correct[contains_mix].cpu().numpy()))
        metrics["language_mix/without_correct_ratio"] = float(np.mean(correct[~contains_mix].cpu().numpy()))
    
        metrics["language_mix/with_reward_mean"] = float(np.mean(reward_tensor[contains_mix].cpu().numpy()))
        metrics["language_mix/without_reward_mean"] = float(np.mean(reward_tensor[~contains_mix].cpu().numpy()))
    
    return metrics


def bootstrap_metric(data: list[dict[str, Any]],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val
