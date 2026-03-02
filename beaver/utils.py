import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import jsonlines


def log_json(data, file_name):
    with open(file_name, "a") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def new_log_dir(log_folder: Path):
    id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    new_log_folder = log_folder / f"logs_{id}"
    assert not os.path.exists(
        new_log_folder
    ), f"Log folder {new_log_folder} already exists"
    os.makedirs(new_log_folder)
    return new_log_folder


def load_jsonl(filename: str) -> List[Dict]:
    """Load JSONL file following LLM-PBE convention."""
    results = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            results.append(obj)
    return results
