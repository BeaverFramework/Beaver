import datetime
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
    with open(filename)  as f:
        for line in f:
            if not line.strip():
                continue
            results.append(json.loads(line))
    return results


def make_histogram(data, bins, max_bar=80):
    counts, edges = np.histogram(data, bins=bins)
    print(f"Total num of elements: {len(data)}")
    print(f"Mean: {np.mean(data)}")
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(counts))]
    max_count = max(counts) if max(counts) > 0 else 1
    print("VR histogram:")
    for i, c in enumerate(counts):
        bar = "█" * int(c / max_count * max_bar)
        print(f"{labels[i]:>10s}\t| {bar} {c}")
