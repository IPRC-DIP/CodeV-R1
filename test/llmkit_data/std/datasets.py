import re

CODE_TEMPLATE = """```python
{}
```
"""

codeblock_pattern = re.compile(r"```python(.+?)```", flags=re.DOTALL)


def extract_code(text: str):
    codes = [match.strip() for match in re.findall(codeblock_pattern, text)]
    if len(codes) > 0:
        code = "\n".join(codes)
        return code
    else:
        return ""


SPLITTER = "__I_wish_it_were_weekends_all_the_time.__"


def detect_dataset_type(item):
    def has_keys(keys):
        return all(map(lambda x: x in item, keys))

    if has_keys(["question", "response"]):
        return "SFT"
    elif has_keys(["prompt", "chosen", "rejected"]):
        return "Reward"
