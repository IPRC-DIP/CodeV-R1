import argparse
import numpy as np

def estimate_pass_k(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, help="Result file", default="codev-qwen32b-o1-prompt-unverified-data-8192-spec-to-rtl-0shot_n20-temp0.6/summary.csv")
    args = parser.parse_args()
    data = np.genfromtxt(args.result_file, delimiter=',')[:, 1:3]
    for k in [1, 5, 10, 20]:
        if k > data[0, 1]: continue
        passk = 0
        for x in data:
            passk += estimate_pass_k(x[1], x[0], k)
        passk /= len(data)
        print(f"Pass@{k} is {passk}.")