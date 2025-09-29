import argparse
from functools import partial

from llmkit_data.std.inference import generate_worker
from llmkit_data.utils.json import read_jsonl, write_jsonl
from llmkit_data.utils.parallel import model_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, help="prompts path")
    parser.add_argument("--out", type=str, help="output path")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--n_sample", type=int, default=10, help="number of samples per task")
    parser.add_argument("--temperature", type=float, default=0.6, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max number of tokens to generate")
    parser.add_argument("--gpu_per_model", type=int, help="Number of GPUs required per model")
    parser.add_argument("--enforce_eager", action="store_true", help="Whether set enforce_eager in vllm")
    args = parser.parse_args()

    import random
    import torch
    import numpy as np
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prompts = list(read_jsonl(args.prompts))
    worker = partial(
        generate_worker,
        model_path=args.model,
        n=int(args.n_sample),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        enforce_eager=args.enforce_eager
    )
    results = model_map(worker, prompts, args.gpu_per_model)

    write_jsonl(results, args.out)
