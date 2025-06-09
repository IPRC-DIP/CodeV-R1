import os

# manual_model_root = "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2"
manual_model_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'test', 'verilogeval-v2')
manual_models = [file_name.split('-')[0] for file_name in os.listdir(manual_model_root) if os.path.isfile(os.path.join(manual_model_root, file_name))]
print("Current manual model list is:", manual_models)
# # manual_models: the model name used in test
# # 每次添加新的测试模型，需要从manual_models和manual_models_datapath添加对应的模型名称和路径
# manual_models = [
#   'manual-rtl-coder',
#   "/lustre/S/zhaoyang/model/RTLCoder-Deepseek-v1.1",
#   "deepseekv2.5",
#   "o1-preview-2024-09-12",
#   "codev-cl",
#   "codev-qw",
#   "codev-dsc",
#   "rtlcoder-dsc",
#   "rtlcoder-dsc-20",
#   "codev-dsc-27k-1",
#   "codev-dsc-27k-20",

#   "codev-cl-1",# 待删去
#   "codev-dsc-1",
#   "codev-cl-20",
#   "codev-dsc-20",
#   "deepseek-coder-6.7b-1",
#   "deepseek-coder-6.7b-20",

#   # 测试师兄的benchmark
#   'codev-qw-20-test1',
#   'codev-qw-20-test2',
#   'codev-qw-20-test3',
#   'gpt-4-turbo-2024-04-09-test1',
#   'gpt-4-turbo-2024-04-09-test2',
#   'gpt-4-turbo-2024-04-09-test3',
#   'RTLCoder-Deepseek-v1.1-test1',
#   'RTLCoder-Deepseek-v1.1-test2',
#   'RTLCoder-Deepseek-v1.1-test3',
#   "Llama-3.1-8B-Instruct-test1",
#   "deepseekv25-test1",
#   "gpt-4o-2024-08-06",
#   "llama-405",
#   "qwq-o1-prompt",
#   "deepseekv3-o1-prompt",
#   "deepseekr1-o1-prompt",
#   "deepseekr1-qwen32b-o1-prompt",
#   "qwencoder-32b-instruct",
#   "codev-qwen32b-o1-prompt",
#   "codev-qwen32b-o1-prompt-8192",
#   "codev-qwen32b-o1-prompt-unverified-data-8192",
# ]
# manual_models_datapath = {
#   "llama-405":[
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/llama405/data/slurm/Llama3.1-405b-instruct-complete-iccad2023_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/llama405/data/slurm/Llama3.1-405b-instruct-complete-iccad2023_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/llama405/data/slurm/Llama3.1-405b-instruct-spec-to-rtl_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/llama405/data/slurm/Llama3.1-405b-instruct-spec-to-rtl_0shot_n1.jsonl",
#   ],
#   "gpt-4o-2024-08-06":[
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4o-2024-08-06/New-gpt-4o-2024-08-06-complete-iccad2023_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4o-2024-08-06/New-gpt-4o-2024-08-06-complete-iccad2023_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4o-2024-08-06/New-gpt-4o-2024-08-06-spec-to-rtl_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4o-2024-08-06/New-gpt-4o-2024-08-06-spec-to-rtl_0shot_n1.jsonl",
#   ],
#   "deepseekv25-test1":[
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/deepseekv25/New-deepseek-v25-complete-iccad2023_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/deepseekv25/New-deepseek-v25-complete-iccad2023_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/deepseekv25/New-deepseek-v25-spec-to-rtl_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/deepseekv25/New-deepseek-v25-spec-to-rtl_0shot_n1.jsonl",
#   ],
#   "Llama-3.1-8B-Instruct-test1":[
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/llama-3.1-8B-Instruct/New-Llama-3.1-8B-Instruct-complete-iccad2023_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/llama-3.1-8B-Instruct/New-Llama-3.1-8B-Instruct-complete-iccad2023_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/llama-3.1-8B-Instruct/New-Llama-3.1-8B-Instruct-spec-to-rtl_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/llama-3.1-8B-Instruct/New-Llama-3.1-8B-Instruct-spec-to-rtl_0shot_n1.jsonl",
#   ],
#   "deepseek-coder-6.7b-1":[
#     "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241031-deepseek-coder-6.7b-instruct-verilogeval_complete-0shot-000/slurm/output0.json",
#     "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241031-deepseek-coder-6.7b-instruct-verilogeval_complete-1shot-000/slurm/output0.json",
#     "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241031-deepseek-coder-6.7b-instruct-verilogeval_spec-0shot-000/slurm/output0.json",
#     "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241031-deepseek-coder-6.7b-instruct-verilogeval_spec-1shot-000/slurm/output0.json",
#   ],
#   "deepseek-coder-6.7b-20":[
#     "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241031-deepseek-coder-6.7b-instruct-verilogeval_complete-0shot-000/slurm/output85.json",
#     "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241031-deepseek-coder-6.7b-instruct-verilogeval_complete-1shot-000/slurm/output85.json",
#     "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241031-deepseek-coder-6.7b-instruct-verilogeval_spec-0shot-000/slurm/output85.json",
#     "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241031-deepseek-coder-6.7b-instruct-verilogeval_spec-1shot-000/slurm/output85.json",
#   ],
#   "codev-qw-20-test1":[
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/codev-qwen-7b/20241116-codev-qwen-7b-verilogeval_spec-0shot-000/slurm/output2.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/codev-qwen-7b/20241116-codev-qwen-7b-verilogeval_spec-0shot-000/slurm/output2.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/codev-qwen-7b/20241116-codev-qwen-7b-verilogeval_spec-0shot-000/slurm/output2.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/codev-qwen-7b/20241116-codev-qwen-7b-verilogeval_spec-0shot-000/slurm/output2.json"
#   ],
#   "codev-qw-20-test2":[
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/codev-qwen-7b/20241114-codev-qwen-7b-verilogeval_complete-0shot-000/slurm/output5.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/codev-qwen-7b/20241114-codev-qwen-7b-verilogeval_complete-0shot-000/slurm/output5.json",
#   ],
#   "codev-qw-20-test3":[
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/codev-qwen-7b/20241114-codev-qwen-7b-verilogeval_complete-0shot-000/slurm/output8.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/codev-qwen-7b/20241114-codev-qwen-7b-verilogeval_complete-0shot-000/slurm/output8.json",
#   ],
#   'gpt-4-turbo-2024-04-09-test1': [
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4-turbo-2024-04-09/verilogeval2/complete-iccad2023_0shot_n1_gpt-4-turbo-2024-04-09-t2-change.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4-turbo-2024-04-09/verilogeval2/complete-iccad2023_0shot_n1_gpt-4-turbo-2024-04-09-t2-change.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4-turbo-2024-04-09/verilogeval2/gpt-4-turbo-2024-04-09-spec-to-rtl_0shot_n1.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4-turbo-2024-04-09/verilogeval2/gpt-4-turbo-2024-04-09-spec-to-rtl_0shot_n1.jsonl"
#   ],
#   'gpt-4-turbo-2024-04-09-test2': [
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4-turbo-2024-04-09/verilogeval2/complete-iccad2023_0shot_n1_gpt-4-turbo-2024-04-09-t5-change.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4-turbo-2024-04-09/verilogeval2/complete-iccad2023_0shot_n1_gpt-4-turbo-2024-04-09-t5-change.jsonl"
#   ],
#   'gpt-4-turbo-2024-04-09-test3': [
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4-turbo-2024-04-09/verilogeval2/complete-iccad2023_0shot_n1_gpt-4-turbo-2024-04-09-t8-change.jsonl",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/gpt-4-turbo-2024-04-09/verilogeval2/complete-iccad2023_0shot_n1_gpt-4-turbo-2024-04-09-t8-change.jsonl"
#   ],
#   'RTLCoder-Deepseek-v1.1-test1': [
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241114-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output2.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241114-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output2.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241116-RTLCoder-Deepseek-v1.1-verilogeval_spec-0shot-000/slurm/output2.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241116-RTLCoder-Deepseek-v1.1-verilogeval_spec-0shot-000/slurm/output2.json"
#   ],
#   'RTLCoder-Deepseek-v1.1-test2': [
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241114-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output5.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241114-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output5.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241114-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output5.json"
#   ],
#   'RTLCoder-Deepseek-v1.1-test3': [
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241114-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output8.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241114-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output8.json",
#     "/workspace/S/zhaoyang/lab/zy2486/llmtest_demo/exps/RTLCoder-Deepseek-v1.1/20241114-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output8.json"
#   ],
#   "deepseekv3-o1-prompt": [
#     None, None,
#     "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2-dsv3-extracted.jsonl",
#     None
#   ],
#   "qwq-o1-prompt": [
#     None, None,
#     "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2-qwq-temp0.6-extracted.jsonl",
#     None
#   ],
#   "deepseekr1-o1-prompt": [
#     None, None,
#     "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2-dsr1-extracted.jsonl",
#     None
#   ],
#   "deepseekr1-qwen32b-o1-prompt": [
#     None, None,
#     "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2-dsr1-qwen-32b-extracted.jsonl",
#     None
#   ],
#   "qwencoder-32b-instruct": [
#     None, None,
#     "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2-qwencoder-32b-extracted.jsonl",
#     None
#   ],
#   "codev-qwen32b-o1-prompt": [
#     None, None,
#     "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2-codev-qwen32b-o1-extracted.jsonl",
#     None
#   ],
#   "codev-qwen32b-o1-prompt-8192": [
#     None, None,
#     "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2-codev-o1-qwen32b-8192-extracted.jsonl",
#     None
#   ],
#   "codev-qwen32b-o1-prompt-unverified-data-8192": [
#     None, None,
#     "/nfs_global/S/zhuyaoyu/projects/CodeV-o1/results/test/verilogeval-v2-codev-o1-qwen32b-unverified-data-8192-extracted.jsonl",
#     None
#   ]
# }
#-------------------------------------------------------------------------
# append function from zy: 20241031-20:01
#-------------------------------------------------------------------------
from openai import OpenAI
def generate_response_ds25(system_msg, full_prompt, response_filename, temperature, top_p, max_tokens):
  # generate from deepseek-coder2.5
  client = OpenAI(api_key="sk-36367a9276af4ec083c25a7ac02ce781", base_url="https://api.deepseek.com")

  response = client.chat.completions.create(
      model="deepseek-chat",
      messages=[
          {"role": "system", "content": system_msg},
          {"role": "user", "content": full_prompt},
      ],
      stream=False,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
  )
  print(response)
  with open(response_filename, 'w') as response_file:
    print(response.choices[0].message.content, file=response_file)
    # print(response.choices[0].message.content, response_file)
  # pass

import json
def generate_response_o1(system_msg, full_prompt, response_filename, output_filename, output_prompt_filename):
  response = ""
  with open(output_filename, 'r', encoding='utf-8') as file_f:
    for line in file_f:
      data = json.loads(line)
      if data['task_id'] == output_prompt_filename:
        response = data['describe']
        break

  print(response)
  with open(response_filename, 'w') as response_file:
    print(response, file=response_file)

def generate_response_codev(system_msg, full_prompt, response_filename, output_filename, output_prompt_filename, number):
  # 从codev数据上获得对应task_id的回复描述
  num = 0
  with open(output_filename, 'r', encoding='utf-8') as file_f:
    for line in file_f:
      data = json.loads(line)
      if data['task_id'] == output_prompt_filename:
        num += 1
        response = data['completion']
        if num == number:
          break

  print(response)
  with open(response_filename, 'w') as response_file:
    print(response, file=response_file)


# generate the response file from model name
def generate_response_file(opts, model, task, system_msg, full_prompt, output_prompt_filename, temperature, top_p, max_tokens):
  if model == "deepseekv2.5":
      response_filename = output_prompt_filename  + "_response.txt"
      generate_response_ds25(system_msg, full_prompt.rstrip(), response_filename, temperature, top_p, max_tokens)
  elif model == "o1-preview-2024-09-12":
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/codev/verilog-eval_v2/verilog-eval/result/o1-preview-2024-09-12_result/complete-iccad2023_1shot_n1_o1-preview-2024-09-12.jsonl"
      else:
        output_filename = "/workspace/S/zhaoyang/codev/verilog-eval_v2/verilog-eval/result/o1-preview-2024-09-12_result/complete-iccad2023_0shot_n1_o1-preview-2024-09-12.jsonl"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/codev/verilog-eval_v2/verilog-eval/result/o1-preview-2024-09-12_result/spec-to-rtl_1shot_n1_o1-preview-2024-09-12.jsonl"
      else:
        output_filename = "/workspace/S/zhaoyang/codev/verilog-eval_v2/verilog-eval/result/o1-preview-2024-09-12_result/spec-to-rtl_0shot_n1_o1-preview-2024-09-12.jsonl"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    generate_response_o1(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question)
  elif "codev-cl" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-cl-7b-verilogeval_complete-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-cl-7b-verilogeval_complete-0shot-000/slurm/output85.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-cl-7b-verilogeval_spec-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-cl-7b-verilogeval_spec-0shot-000/slurm/output85.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "codev-qw" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-qwen-7b-verilogeval_complete-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-qwen-7b-verilogeval_complete-0shot-000/slurm/output85.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-qwen-7b-verilogeval_spec-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-qwen-7b-verilogeval_spec-0shot-000/slurm/output85.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "codev-dsc" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-dsc-6.7b-verilogeval_complete-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-dsc-6.7b-verilogeval_complete-0shot-000/slurm/output85.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-dsc-6.7b-verilogeval_spec-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/codev-dsc-6.7b-verilogeval_spec-0shot-000/slurm/output85.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "rtlcoder-dsc" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-RTLCoder-Deepseek-v1.1-verilogeval_complete-1shot-000/slurm/output0.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output0.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-RTLCoder-Deepseek-v1.1-verilogeval_spec-1shot-000/slurm/output0.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-RTLCoder-Deepseek-v1.1-verilogeval_spec-0shot-000/slurm/output0.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "rtlcoder-dsc-20" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-RTLCoder-Deepseek-v1.1-verilogeval_complete-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-RTLCoder-Deepseek-v1.1-verilogeval_complete-0shot-000/slurm/output85.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-RTLCoder-Deepseek-v1.1-verilogeval_spec-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-RTLCoder-Deepseek-v1.1-verilogeval_spec-0shot-000/slurm/output85.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "codev-dsc-27k-1" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-roughl-gpt35-27k-20240520-nlH2v-dsc-verilogeval_complete-1shot-000/slurm/output0.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-roughl-gpt35-27k-20240520-nlH2v-dsc-verilogeval_complete-0shot-000/slurm/output0.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-roughl-gpt35-27k-20240520-nlH2v-dsc-verilogeval_spec-1shot-000/slurm/output0.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-roughl-gpt35-27k-20240520-nlH2v-dsc-verilogeval_spec-0shot-000/slurm/output0.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "codev-dsc-27k-20" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-roughl-gpt35-27k-20240520-nlH2v-dsc-verilogeval_complete-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-roughl-gpt35-27k-20240520-nlH2v-dsc-verilogeval_complete-0shot-000/slurm/output85.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-roughl-gpt35-27k-20240520-nlH2v-dsc-verilogeval_spec-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-roughl-gpt35-27k-20240520-nlH2v-dsc-verilogeval_spec-0shot-000/slurm/output85.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "codev-dsc-1" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-dsc-6.7b-verilogeval_complete-1shot-000/slurm/output0.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-dsc-6.7b-verilogeval_complete-0shot-000/slurm/output0.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-dsc-6.7b-verilogeval_spec-1shot-000/slurm/output0.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-dsc-6.7b-verilogeval_spec-0shot-000/slurm/output0.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "codev-dsc-20" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-dsc-6.7b-verilogeval_complete-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-dsc-6.7b-verilogeval_complete-0shot-000/slurm/output85.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-dsc-6.7b-verilogeval_spec-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-dsc-6.7b-verilogeval_spec-0shot-000/slurm/output85.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "codev-cl-1" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-cl-7b-verilogeval_complete-1shot-000/slurm/output0.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-cl-7b-verilogeval_complete-0shot-000/slurm/output0.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-cl-7b-verilogeval_spec-1shot-000/slurm/output0.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-cl-7b-verilogeval_spec-0shot-000/slurm/output0.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  elif "codev-cl-20" == model:
    if task == "code-complete-iccad2023":
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-cl-7b-verilogeval_complete-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-cl-7b-verilogeval_complete-0shot-000/slurm/output85.json"
    else:
      if opts.examples:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-cl-7b-verilogeval_spec-1shot-000/slurm/output85.json"
      else:
        output_filename = "/workspace/S/zhaoyang/lab/llmtest_demo/exps/20241030-codev-cl-7b-verilogeval_spec-0shot-000/slurm/output85.json"
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)
  else:
    # if task == "code-complete-iccad2023":
    #   if not opts.examples:
    #     output_filename = manual_models_datapath[model][0]
    #   else:
    #     output_filename = manual_models_datapath[model][1]
    # else:
    #   if not opts.examples:
    #     output_filename = manual_models_datapath[model][2]
    #   else:
    #     output_filename = manual_models_datapath[model][3]
    # task = "complete" if task == "code-complete-iccad2023" else "spec_to_rtl"
    task = task.replace("-", "_")
    shot = 0 if not opts.examples else 1
    output_filename = os.path.join(manual_model_root, f"{model}-{task}-shot_{shot}-temp_{temperature}.jsonl")
    if not os.path.exists(output_filename) and int(temperature) == temperature:
      output_filename = os.path.join(manual_model_root, f"{model}-{task}-shot_{shot}-temp_{int(temperature)}.jsonl")
    
    # 测试o1的效果
    response_filename = output_prompt_filename  + "_response.txt"
    output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
    output_prompt_filename_question_num = int(output_prompt_filename[-2:])
    generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)