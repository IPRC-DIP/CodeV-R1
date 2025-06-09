import os
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, combinations
import subprocess
import numpy as np


def get_distance(connection_type):
    if connection_type.startswith("NV"):
        return 1
    elif connection_type == "X":
        return 0
    elif connection_type == "PIX":
        return 2
    elif connection_type == "PXB":
        return 3
    elif connection_type == "PHB":
        return 4
    elif connection_type == "NODE":
        return 5
    elif connection_type == "SYS":
        return 6
    else:
        raise RuntimeError(f"Unknown connection type {connection_type}")


def get_gpu_topology():
    """
    Get the GPU topology using `nvidia-smi topo -m` and return a distance matrix.
    """
    try:
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], stdout=subprocess.PIPE, text=True)
        topo_output = result.stdout
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. Make sure NVIDIA drivers are installed and nvidia-smi is in PATH.")

    # Parse the topology matrix
    matrix_str = topo_output.split('\n\n')[0]
    lines = matrix_str.splitlines()

    header = lines[0].split()
    gpu_num = sum([x.startswith("GPU") for x in header])

    for idx in range(gpu_num):
        assert header[idx].endswith(f"GPU{idx}"), header[idx]

    matrix = []
    for idx, line in enumerate(lines[1:1 + gpu_num]):
        assert line.startswith(f"GPU{idx}")
        matrix.append(line.split()[1:1 + gpu_num])

    # Convert to a numeric distance matrix (lower is better)
    distance_matrix = [[get_distance(e) for e in r] for r in matrix]
    return np.array(distance_matrix)


def comb_group(n, k):
    groups = []

    def helper(lst):
        if len(lst) == 0:
            yield groups.copy()
        else:
            head, *rest = lst
            for group in combinations(rest, k-1):
                groups.append((head,) + group)
                yield from helper([x for x in rest if x not in group])
                groups.pop()

    yield from helper(list(range(n)))


def allocate_gpu(model_required_gpus):
    cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')

    gpu_num = len(cuda_devices)
    assert gpu_num % model_required_gpus == 0, "gpus must be n * tensor_parallel"

    gpu_ids = [int(x) for x in cuda_devices]
    m = get_gpu_topology()[gpu_ids][:, gpu_ids]

    cost_memory = dict()
    for group in combinations(range(gpu_num), model_required_gpus):
        indices = list(group)
        cost_memory[group] = np.sum(m[indices][:, indices])

    min_cost, min_groups = float('inf'), []
    for groups in comb_group(gpu_num, model_required_gpus):
        cost = sum(cost_memory[group] for group in groups)
        if cost < min_cost:
            min_cost, min_groups = cost, groups

    return [[str(gpu_ids[x]) for x in group] for group in min_groups]


def split_data(data, num):
    """
    The average length of chat in the dataset is not uniformly distributed.
    Sometimes, the initial chats are shorter, while the later ones are longer.
    To ensure that all GPUs have nearly the same execution time,
    we intentionally shuffle the dataset.
    """
    groups = [[] for _ in range(num)]
    for i, item in enumerate(data):
        item["__index__"] = i
        groups[i % num].append(item)
    return groups


def sort_data(lst):
    lst.sort(key=lambda x: x["__index__"])
    for item in lst:
        item.pop("__index__")
    return lst


def model_map(worker, data, model_required_gpus):
    cuda_devices = allocate_gpu(model_required_gpus)
    group_num = len(cuda_devices)
    data_groups = split_data(data, group_num)

    args = list(zip(cuda_devices, data_groups))
    with ProcessPoolExecutor() as executor:
        nested_results = list(executor.map(worker, *zip(*args))) # It's a magic

    return sort_data(list(chain(*nested_results)))
