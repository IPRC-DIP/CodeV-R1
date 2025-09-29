import os
import time
import tqdm
from scipy.special import comb
from typing import List, Union, Iterable, Dict, Tuple, Optional
import numpy as np
import itertools
from threading import Thread
import argparse
import statistics
import shutil
import contextlib
from multiprocessing import Pool, Manager

def copy_folder(src, dst):
    try:
        if os.path.exists(dst):
            shutil.rmtree(dst)
            print(f"Removed existing directory {dst}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)
        print(f"Successfully copied {src} to {dst}")
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")

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
    sum_list = []
    for design in dic_list.keys():
        c = dic_list[design]["syntax_success"]
        sum_list.append(1 - comb(n - c, k) / comb(n, k))
    syntax_passk = sum(sum_list) / len(sum_list)

    sum_list = []
    for design in dic_list.keys():
        c = dic_list[design]["func_success"]
        sum_list.append(1 - comb(n - c, k) / comb(n, k))
    func_passk = sum(sum_list) / len(sum_list)
    print(f"syntax pass@{k}: {syntax_passk},   func pass@{k}: {func_passk}")

def process_design(design, testfile, path, test_path, current_path):
    design_path = os.path.join(test_path, design)
    design_path_makefile = os.path.join(design_path, "makefile")
    result = {"syntax_success": 0, "func_success": 0}
    result_t = {"syntax_success": 0, "func_success": 0}

    if os.path.exists(design_path_makefile):
        with open(design_path_makefile, "r") as file:
            makefile_content = file.read()
            modified_makefile_content = makefile_content.replace(
                "${TEST_DESIGN}", f"{path}/{testfile}/{design}"
            )
        with open(design_path_makefile, "w") as file:
            file.write(modified_makefile_content)
        os.chdir(design_path)
        os.system("timeout 5m make vcs")
        simv_generated = False
        if os.path.exists("simv"):
            simv_generated = True
        if simv_generated:
            result["syntax_success"] += 1
            result_t["syntax_success"] = 1
            to_flag = exec_shell("timeout 5m make sim > output.txt")
            if to_flag == 1:
                with open("output.txt", "r") as file:
                    output = file.read()
                    if "Pass" in output or "pass" in output:
                        result["func_success"] += 1
                        result_t["func_success"] = 1
                    else:
                        result_t["func_success"] = 0
            else:
                result_t["func_success"] = 0
        else:
            result_t["syntax_success"] = 0
            result_t["func_success"] = 0
        with open("makefile", "w") as file:
            file.write(makefile_content)
        os.system("make clean")
        os.chdir(current_path)
    return design, result, result_t

def test_one_file(testfile, result_dic, path, progress_bar, test_path):
    current_path = os.getcwd()
    print(f"Current working directory: {current_path}")
    design_name = ['alu', 'multi_pipe_4bit', 'multi_booth_8bit', 'calendar', 'traffic_light', 'edge_detect', 'RAM', 'JC_counter', 'fsm', 'serial2parallel', 'adder_8bit', 'asyn_fifo', 'multi_pipe_8bit', 'multi_16bit', 'radix2_div', 'right_shifter', 'adder_pipe_64bit', 'div_16bit', 'freq_div', 'counter_12', 'signal_generator', 'pulse_detect', 'parallel2serial', 'adder_16bit', 'pe', 'adder_32bit', 'synchronizer', 'accu', 'width_8to16']
    print(testfile)

    with Pool() as pool:
        results = pool.starmap(process_design, [(design, testfile, path, test_path, current_path) for design in design_name])

    for design, result, result_t in results:
        result_dic[design]["syntax_success"] += result["syntax_success"]
        result_dic[design]["func_success"] += result["func_success"]
        result_dic_t[design]["syntax_success"].append(result_t["syntax_success"])
        result_dic_t[design]["func_success"].append(result_t["func_success"])
        progress_bar.update(1)

    return result_dic

def main(args):
    path = args.path
    design_name = ['alu', 'multi_pipe_4bit', 'multi_booth_8bit', 'calendar', 'traffic_light', 'edge_detect', 'RAM', 'JC_counter', 'fsm', 'serial2parallel', 'adder_8bit', 'asyn_fifo', 'multi_pipe_8bit', 'multi_16bit', 'radix2_div', 'right_shifter', 'adder_pipe_64bit', 'div_16bit', 'freq_div', 'counter_12', 'signal_generator', 'pulse_detect', 'parallel2serial', 'adder_16bit', 'pe', 'adder_32bit', 'synchronizer', 'accu', 'width_8to16']
    
    test_path = os.path.join(args.test_dir,"RTLLM")

    copy_folder("Original_problem", test_path)

    result_dic = {key: {"syntax_success": 0, "func_success": 0} for key in design_name}
    global result_dic_t
    result_dic_t = {key: {"syntax_success": [], "func_success": []} for key in design_name}

    progress_bar = tqdm.tqdm(total=args.n*29)
    file_id = 1
    n = 0
    while os.path.exists(os.path.join(path, f"t{file_id}")):
        result_dic = test_one_file(f"t{file_id}", result_dic, path, progress_bar, test_path)
        n += 1
        file_id += 1

    print(result_dic)
    sk = [1, 5, 10]
    for k in sk:
        cal_atk(result_dic, n, k)

    total_syntax_success = sum(1 for item in design_name if result_dic[item]["syntax_success"] != 0)
    total_func_success = sum(1 for item in design_name if result_dic[item]["func_success"] != 0)

    print(f"total_syntax_success: {total_syntax_success}/{len(design_name)}")
    print(f"syntax pass rate = {total_syntax_success / len(design_name)}")
    print(f"total_func_success: {total_func_success}/{len(design_name)}")
    print(f"func pass rate = {total_func_success / len(design_name)}")

    if args.n == 20:
        print("*************采样计算*************************8")
        elements = list(range(20))
        combinations = list(itertools.combinations(elements, 5))
        print("总组合数:", len(combinations))
        print(result_dic_t)
        syntax_success = []
        func_success = []
        for combo in combinations:
            total_syntax_success = 0
            total_func_success = 0
            for q in design_name:
                x = sum(1 for item in combo if result_dic_t[q]["syntax_success"][item] != 0)
                y = sum(1 for item in combo if result_dic_t[q]["func_success"][item] != 0)
                if x != 0:
                    total_syntax_success += 1
                if y != 0:
                    total_func_success += 1
            syntax_success.append(total_syntax_success / len(design_name))
            func_success.append(total_func_success / len(design_name))
        syntax_mean = statistics.mean(syntax_success)
        syntax_std_dev = statistics.stdev(syntax_success)
        func_mean = statistics.mean(func_success)
        func_std_dev = statistics.stdev(func_success)
        print(f"syntax_mean: {syntax_mean}")
        print(f"syntax_std_dev: {syntax_std_dev}")
        print(f"func_mean: {func_mean}")
        print(f"func_std_dev: {func_std_dev}")
    print(f"result_dic\n{result_dic}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--n', type=int, default=20)
    args = parser.parse_args()
    with open(f"{args.test_dir}/result.txt", "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            main(args)
