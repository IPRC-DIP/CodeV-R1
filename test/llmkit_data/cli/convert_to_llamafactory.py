import argparse

from llmkit_data.utils.json import read_jsonl
from llmkit_data.std.datasets import detect_dataset_type
from llmkit_data.converter.llamafactory import (
    stdsft_to_llamafactory,
    stdreward_to_llamafactory,
    mk_sft_dataset_info,
    mk_reward_dataset_info,
    save_dataset
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset path")
    parser.add_argument("--llamafactory", type=str, help="llamafactory path")
    parser.add_argument("--dataset_name", type=str, help="new dataset name")
    args = parser.parse_args()

    dataset = list(read_jsonl(args.dataset))

    match detect_dataset_type(dataset[0]):
        case "SFT":
            new_dataset_info = mk_sft_dataset_info(args.dataset_name)
            new_dataset = stdsft_to_llamafactory(dataset)
        case "Reward":
            new_dataset_info = mk_reward_dataset_info(args.dataset_name)
            new_dataset = stdreward_to_llamafactory(dataset)
        case other:
            raise NotImplementedError("Unknown dataset")

    save_dataset(args.llamafactory, new_dataset_info, new_dataset)
