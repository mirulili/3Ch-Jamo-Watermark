import json
import os
import random
from dataclasses import dataclass

import torch


@dataclass
class ExperimentConfig:
    model_name: str = "skt/kogpt2-base-v2"
    mode: str = "robustness"
    k_bits: int = 2
    top_k: int = 40
    bias_value: float = 5.0
    max_length: int = 400
    seed: int = 42
    # Optional path to a frequency-balanced LUT produced by corpus_analysis.
    # When None, the hash policy falls back to modulo bucketing.
    lut_path: str | None = os.path.join(
        os.path.dirname(__file__), "..", "results", "hash_lut.json"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lut(path: str | None) -> dict[str, dict[int, int]] | None:
    """
    Load a hash LUT from `path` and return the runtime dict form
    `{channel: {jamo_idx: bucket_idx}}` with int keys. Returns None when the
    path is None or the file is missing — callers should treat that as
    "fall back to modulo bucketing".
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("lut", {})
    return {
        channel: {int(k): int(v) for k, v in mapping.items()}
        for channel, mapping in raw.items()
    }
