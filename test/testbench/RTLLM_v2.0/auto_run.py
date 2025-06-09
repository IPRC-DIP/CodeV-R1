import os
import time
import tqdm
from scipy.special import comb

#import threading
from threading import Thread


def exec_shell(cmd_str, timeout=8):
    def run_shell_func(sh):
        os.system(sh)
    start_time = time.time()
    t = Thread(target=run_shell_func, args=(cmd_str,), daemon=False)
    t.start()
    while 1:
        now = time.time()
        if now - start_time >= timeout:
            if not t.is_alive():
                return 1
            else:
                return 0
        if not t.is_alive():
            return 1
        time.sleep(1)


def cal_atk(dic_list, n, k):
    #syntax 
    sum_list = []
    for design in dic_list.keys():
        c = dic_list[design]['syntax_success']
        sum_list.append(1 - comb(n - c, k) / comb(n, k))
    sum_list.append(0)
    syntax_passk = sum(sum_list) / len(sum_list)
    
    #func
    sum_list = []
    for design in dic_list.keys():
        c = dic_list[design]['func_success']
        sum_list.append(1 - comb(n - c, k) / comb(n, k))
    sum_list.append(0)
    func_passk = sum(sum_list) / len(sum_list)
    print(f'syntax pass@{k}: {syntax_passk},   func pass@{k}: {func_passk}')


progress_bar = tqdm.tqdm(total=1000)
design_name = ['up_down_counter', 'counter_12', 'JC_counter', 'ring_counter', 'fsm', 'sequence_detector', 'LFSR', 'right_shifter', 'barrel_shifter', 'asyn_fifo', 'LIFObuffer', 'parallel2serial', 'pulse_detect', 'traffic_light', 'edge_detect', 'synchronizer', 'width_8to16', 'calendar', 'serial2parallel', 'square_wave', 'signal_generator', 'RAM', 'alu', 'instr_reg', 'ROM', 'clkgenerator', 'pe', 'freq_div', 'freq_divbyeven', 'freq_divbyodd', 'freq_divbyfrac', 'radix2_div', 'div_16bit', 'comparator_3bit', 'comparator_4bit', 'multi_8bit', 'multi_pipe_4bit', 'multi_booth_8bit', 'multi_16bit', 'multi_pipe_8bit', 'adder_bcd', 'adder_pipe_64bit', 'adder_16bit', 'adder_8bit', 'adder_32bit', 'accu', 'fixed_point_adder', 'fixed_point_substractor', 'float_multi', 'sub_64bit']
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--jsonloutpath', type=str)
args = parser.parse_args()
path = args.path

result_dic = {key: {} for key in design_name}
for item in design_name:
    result_dic[item]['syntax_success'] = 0
    result_dic[item]['func_success'] = 0

all_data = []

def test_one_file(testfile, result_dic):
    for design in design_name:
        task_id = design
        syntax = 0
        func = 0
        code = ""
        gloden_code = ""
        print(design)
        verified_files = [f for f in os.listdir(design) if f.startswith("verified_")]
        if verified_files:
            with open(os.path.join(design, verified_files[0]), "r") as file:
                gloden_code = file.read()

        print(f"{design}/makefile")
        if os.path.exists(f"{design}/makefile"):
            makefile_path = os.path.join(design, "makefile")
            code_file = f"{path}/{testfile}/{design}.v"
            if os.path.exists(code_file):
                with open(code_file, "r") as file:
                    code = file.read()
            with open(makefile_path, "r") as file:
                makefile_content = file.read()
                modified_makefile_content = makefile_content.replace("${TEST_DESIGN}", f"{path}/{testfile}/{design}")
                # modified_makefile_content = makefile_content.replace(f"{path}/{design}/{design}", "${TEST_DESIGN}")
            with open(makefile_path, "w") as file:
                file.write(modified_makefile_content)
            print("*"*100)
            # Run 'make vcs' in the design folder
            os.chdir(design)
            os.system("timeout 5m  make vcs")
            simv_generated = False
            if os.path.exists("simv"):
                simv_generated = True


            if simv_generated:
                result_dic[design]['syntax_success'] += 1
                syntax = 1
                # Run 'make sim' and check the result
                #os.system("make sim > output.txt")
                print("-"*100)
                to_flag = exec_shell("timeout 5m  make sim > output.txt")
                if to_flag == 1:
                    with open("output.txt", "r") as file:
                        output = file.read()
                        if "Pass" in output or "pass" in output:
                            result_dic[design]['func_success'] += 1
                            func = 1
            
            with open("makefile", "w") as file:
                file.write(makefile_content)
            os.system("make clean")
            os.chdir("..")
            progress_bar.update(1)
            all_data.append({"task_id":task_id, "code":code, "syntax":syntax, "semantic":func, "formal":2, "ref_code": gloden_code})

    return result_dic

file_id = 1
n = 0
while os.path.exists(os.path.join(path, f"t{file_id}")):
    # if file_id == 5:
    #     break
    result_dic = test_one_file(f"t{file_id}", result_dic)
    n += 1
    file_id += 1

print(result_dic)
sk = [1, 5,10]
for k in sk:
    cal_atk(result_dic, n, k)

total_syntax_success = 0
total_func_success = 0

# 对于结果进行分析
for item in design_name:
    if result_dic[item]['syntax_success'] != 0:
        total_syntax_success += 1
    if result_dic[item]['func_success'] != 0:
        total_func_success += 1
print(f'total_syntax_success: {total_syntax_success}/{len(design_name)}')
print(f'total_func_success: {total_func_success}/{len(design_name)}')
# print(f"Syntax Success: {syntax_success}/{len(design_name)}")
# print(f"Functional Success: {func_success}/{len(design_name)}")
with open(args.jsonloutpath, 'w') as jsonl_file:
    for entry in all_data:
        jsonl_file.write(json.dumps(entry) + '\n')