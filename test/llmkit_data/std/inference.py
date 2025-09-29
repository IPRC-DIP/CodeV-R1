import os
import numpy as np
from vllm import LLM, SamplingParams
from llmkit_data.std.datasets import SPLITTER


def generate_worker(cuda_device, prompts, model_path, n, temperature, max_tokens, top_p, enforce_eager):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_device)
    print("enforce eager is", enforce_eager)

    llm = LLM(
        model=model_path,
        seed=42,
        max_model_len=max(4096, max_tokens),
        max_seq_len_to_capture=max(4096, max_tokens),
        swap_space=32,
        tensor_parallel_size=len(cuda_device),
        enforce_eager=enforce_eager,
    )

    tokenizer = llm.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id]
    print(f"SUCCESS: load llm {model_path} on cuda {cuda_device}")
    print("Temperature is", temperature)

    vllm_sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        seed=42
    )

    def messages_to_text(messages):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if SPLITTER in text:
            text = text.split(SPLITTER)[0]
        return text

    text_prompts = [messages_to_text(item["question"]) for item in prompts]

    outputs = llm.generate(
        text_prompts, sampling_params=vllm_sampling_params, use_tqdm=True
    )

    results = []
    for item, output in zip(prompts, outputs):
        messages = item["question"]
        if SPLITTER in messages[-1]["content"]:
            slot_message = messages.pop()
            raw_content = slot_message["content"].split(SPLITTER)[0]
            raw_role = slot_message["role"]
            has_splitter = True
        else:
            has_splitter = False

        for response in output.outputs:
            generated_text = response.text

            if has_splitter:
                message["content"] = raw_content + generated_text
                message = {"role": raw_role, "content": raw_content + generated_text}
            else:
                message = {"role": "assistant", "content": generated_text}

            results.append({**item, "response": [message]})

    return results
