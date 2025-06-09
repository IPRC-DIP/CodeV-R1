from llmkit_data.utils.json import read_json, write_json


def mk_sft_dataset_info(dataset_name):
    return {
        dataset_name: {
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def mk_sft_item(messages):
    return {"messages": messages}


def mk_reward_dataset_info(dataset_name):
    return {
        dataset_name: {
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "messages",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def mk_reward_item(prompt, chosen, rejected):
    return {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
    }


def save_dataset(llamafactory_path, dataset_info, dataset):
    all_dataset_info_path = f"{llamafactory_path}/data/dataset_info.json"
    all_dataset_info = read_json(all_dataset_info_path)
    all_dataset_info |= dataset_info
    write_json(all_dataset_info, all_dataset_info_path, indent=4)

    assert len(dataset_info.keys()) == 1
    dataset_name = list(dataset_info.keys())[0]
    dataset_relative_path = dataset_info[dataset_name]["file_name"]
    write_json(list(dataset), f"{llamafactory_path}/data/{dataset_relative_path}")


def stdsft_to_llamafactory(dataset):
    for item in dataset:
        yield mk_sft_item(item["question"] + item["response"])


def stdreward_to_llamafactory(dataset):
    for item in dataset:
        assert (
            len(item["chosen"]) == 1 and len(item["rejected"]) == 1
        ), "llamafactory only support 1 turn response"
        yield mk_reward_item(item["prompt"], item["chosen"]["content"], item["rejected"]["content"])
