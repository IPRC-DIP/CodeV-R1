import json
from pathlib import Path

def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

def read_jsonl(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    except GeneratorExit:
        print("Generator was closed")


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(data, file_path):
    ensure_parent(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def write_json(data, file_path, indent=None):
    ensure_parent(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
