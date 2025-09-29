import os

manual_model_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'test', 'verilogeval-v2')
manual_models = [file_name.split('-')[0] for file_name in os.listdir(manual_model_root) if os.path.isfile(os.path.join(manual_model_root, file_name))]
print("Current manual model list is:", manual_models)

from openai import OpenAI

import json

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
  
  response_filename = output_prompt_filename  + "_response.txt"
  output_prompt_filename_question = output_prompt_filename[:output_prompt_filename.find("/")]
  output_prompt_filename_question_num = int(output_prompt_filename[-2:])
  generate_response_codev(system_msg, full_prompt.rstrip(), response_filename, output_filename, output_prompt_filename_question, output_prompt_filename_question_num)