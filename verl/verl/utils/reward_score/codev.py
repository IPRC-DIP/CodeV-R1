import re
import pickle
from verl.utils.reward_score.codev_eval_toolkit.eval_codev import verify_one_sample, verify_one_sample_wrapper, extract_verilog


def compute_score(solution_str, ground_truth, **kwargs):
    gts = pickle.loads(ground_truth)
    
    response_pos = solution_str.find("<|im_start|>assistant")
    if response_pos >= 0:
        solution_str = solution_str[response_pos:]
    else:
        pass

    def check_format(output):
        tags = ["<think>", "</think>", "<answer>", "</answer>"]
        tag_count = [output.count(tag) for tag in tags]
        positions = [output.find(tag) for tag in tags]
        return min(tag_count) == max(tag_count) == 1 and positions[0] < positions[1] < positions[2] < positions[3]

    def calc_reward(solution_str, ground_truth):
        extracted_answer = extract_verilog(solution_str)
        if not check_format(solution_str) or extracted_answer is None:
            reward = 0.0
        else:
            result = verify_one_sample_wrapper((ground_truth, extracted_answer))
            if result["correct"] == True:
                reward = 1.0
            else:
                reward = 0.0
        return reward

    rewards = [calc_reward(solution_str, gt) for gt in gts.values()]
    reward = max(rewards)

    return reward


def compute_score_wrapper(data_source, solution_str, ground_truth, extra_info, **kwargs):
    return compute_score(solution_str, ground_truth, **kwargs)


if __name__ == '__main__':
    file = "data/codev/v1/3.1k_r1_filtered/train.parquet"
    import pyarrow.parquet as pq
    data = pq.read_table(file).to_pylist()
    
    sep = "============================================"
    print(data[0].keys())
    # correct
    gt = data[0]['reward_model']['ground_truth']
    example_ans = pickle.loads(gt)['answer']['code']
    example_output = f"<think></think>  <answer>\n```verilog\n{example_ans}```\n</answer>"
    reward = compute_score(example_output, gt)
    print(f"{sep}\n{example_output}\n{sep}\n{reward}")

    # wrong format
    example_output = f"<think> <answer></think> ```verilog\n{example_ans}```</answer>"
    reward = compute_score(example_output, gt)
    print(f"{sep}\n{example_output}\n{sep}\n{reward}")

    # wrong answer
    example_output = f"<think> </think> <answer>\n```verilog\n```\n</answer>"
    reward = compute_score(example_output, gt)
    print(f"{sep}\n{example_output}\n{sep}\n{reward}")