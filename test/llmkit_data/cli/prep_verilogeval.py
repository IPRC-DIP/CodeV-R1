import os
from pathlib import Path
import json
import argparse
from llmkit_data.utils.json import read_jsonl, write_jsonl


def mk_prompt_o1_training(question):
    question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")
    prompt = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.

Please structure your response into two main sections: Thought and Solution.

In the Thought section, detail your reasoning process using the specified format:
```
<|begin_of_thought|>
{thought with steps separated with "\n\n"}
<|end_of_thought|>
```
Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.

In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:
```
<|begin_of_solution|>
{final formatted, precise, and clear solution}
<|end_of_solution|>
```
Now, try to write the corresponding verilog code based on the following content through the above guidelines:
"""

    prompt += question

    conversation = [{"role": "user", "content": prompt}]
    return conversation


def mk_prompt_o1_general(question):
    question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")
    prompt = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.

Please structure your response into two main sections: Thought and Solution.

In the Thought section, the thinking steps should be split by "\n\n". Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.

In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion.
Now, try to write the corresponding verilog code based on the following content through the above guidelines:
"""

    prompt += question

    conversation = [{"role": "user", "content": prompt}]
    return conversation


def mk_prompt_r1_v1(question):
    
    question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")

    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    user_prompt = "Now, try to write the corresponding verilog code based on the following content:\n" + question

    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return conversation


def mk_prompt_r1(question):
    
    question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")
    
    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    user_prompt = "Now, try to write the corresponding verilog code based on the following content:\n" + question
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return conversation


def mk_prompt_r1_user(question):
    
    question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")

    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    user_prompt = "Now, try to write the corresponding verilog code based on the following content:\n" + question

    conversation = [{"role": "user", "content": system_prompt + user_prompt}]
    return conversation


def mk_prompt_codev_rl(question):
    question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")
    system_prompt = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to write verilog code. After thinking, when you finally reach a conclusion, enclose the final verilog code in ```verilog ``` within <answer> </answer> tags. i.e., <answer> ```verilog\n module top_module(in, out, ...); ... ``` </answer>.\n"""
    user_prompt = question.strip() + "\n"
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return conversation


def convert_to_sft_v1(data_path, description_path, prompt_type):
    tasks = list(read_jsonl(data_path))
    descriptions = list(read_jsonl(description_path))
    tasks.sort(key=lambda item: item["task_id"])
    descriptions.sort(key=lambda item: item["task_id"])
    for task, description in zip(tasks, descriptions):
        question = description['detail_description'] + "The module head should be:\n" + task['prompt']
        if prompt_type == "simple":
            question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")
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


def convert_to_sft_v2(data_path, prompt_type):
    for task in read_jsonl(data_path):
        question = task["fullprompt"]
        if question.count('//') >= 2 and 'module TopModule' in question:
            # newly added
            question = question.split('//')
            question[-1] = "The module head should be:" + question[-1]
            question = "".join(question)
            # question = "```verilog\n" + question + "```"
        if prompt_type == "simple":
            question = question.replace("Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n", "")
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
    parser.add_argument("--version", type=str, default="v2", help="verilog-eval version", choices=["v1", "v2"])
    parser.add_argument("--data_path", type=str, help="verilog-eval data path")
    parser.add_argument("--description_path", type=str, help="verilog-eval description path")
    parser.add_argument("--out", type=str, help="output path")
    parser.add_argument("--prompt_type", type=str, choices=["simple", "o1_general", "o1_training", "r1", "codev_rl"], help="One of `simple`, `o1_general`, `o1_training`, `r1`, `codev_rl`")
    args = parser.parse_args()

    if args.version == "v1":
        write_jsonl(convert_to_sft_v1(args.data_path, args.description_path, args.prompt_type), args.out)
    else:
        write_jsonl(convert_to_sft_v2(args.data_path, args.prompt_type), args.out)