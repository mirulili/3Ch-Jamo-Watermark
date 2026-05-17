"""
Phase 1 corpus analysis: characterizes the KoGPT2 (or any HuggingFace) vocabulary
from a Jamo-watermarking perspective.

Outputs:
  - Printed summary table
  - results/corpus_analysis.json
  - results/corpus_analysis_heatmap.png  (per-channel × per-bucket counts)
"""
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from ..watermark.hash_policy import HashPolicy
from ..watermark.jamo_utils import get_last_syllable_jamo


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
HASH_LUT_PATH = os.path.join(RESULTS_DIR, "hash_lut.json")
# A bucket is considered "starved" if its share of watermarkable tokens is
# below this fraction of uniform expectation. With k_bits=2 a uniform bucket
# would hold 25% of tokens; at threshold 0.5 a bucket below 12.5% counts as
# starved and is a likely contributor to stuck-state runs.
STARVED_THRESHOLD = 0.5
# Number of jamo values per channel.
JAMO_RANGES = (19, 21, 28)  # choseong, jungseong, jongseong
CHANNEL_NAMES = ("choseong", "jungseong", "jongseong")


def build_balanced_lut(
    jamo_data: list[tuple[int, int, int]],
    k_bits: int,
) -> dict:
    """
    Constructs a per-channel jamo_index → bucket_index LUT that balances the
    total vocabulary mass across the 2^k_bits hash buckets.

    Greedy algorithm: for each channel, sort jamo indices by vocabulary count
    (descending, ties broken by index ascending for determinism). Assign each
    index to whichever bucket currently has the smallest cumulative count.

    Returns a dict shaped like:
        {
          "buckets": 2 ** k_bits,
          "lut": {
            "choseong":  {jamo_idx (int): bucket_idx (int), ...},
            "jungseong": {...},
            "jongseong": {...},
          },
          "bucket_sizes": {channel_name: [size per bucket, ...]},
        }
    """
    buckets = 2 ** k_bits
    lut: dict[str, dict[int, int]] = {}
    bucket_sizes_out: dict[str, list[int]] = {}

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        max_idx = JAMO_RANGES[ch_idx]
        index_counts: Counter = Counter()
        for triple in jamo_data:
            index_counts[triple[ch_idx]] += 1
        # Include zero-count indices so the LUT is total over [0, max_idx).
        for i in range(max_idx):
            index_counts.setdefault(i, 0)

        bucket_sizes = [0] * buckets
        ch_lut: dict[int, int] = {}
        sorted_indices = sorted(index_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for idx, count in sorted_indices:
            target_bucket = min(range(buckets), key=lambda b: bucket_sizes[b])
            ch_lut[idx] = target_bucket
            bucket_sizes[target_bucket] += count

        lut[ch_name] = ch_lut
        bucket_sizes_out[ch_name] = bucket_sizes

    return {
        "buckets": buckets,
        "lut": lut,
        "bucket_sizes": bucket_sizes_out,
    }


def load_hash_lut(path: str = HASH_LUT_PATH) -> dict[str, dict[int, int]] | None:
    """
    Load a hash LUT JSON written by `run_analysis`. Returns the inner
    `{channel: {jamo_idx: bucket_idx}}` mapping with int keys, or None if the
    file doesn't exist. HashPolicy accepts this directly.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("lut", {})
    return {
        channel: {int(k): int(v) for k, v in mapping.items()}
        for channel, mapping in raw.items()
    }


def analyze_vocabulary(model_name: str = "skt/kogpt2-base-v2") -> dict:
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    jamo_data: list[tuple[int, int, int]] = []
    no_jongseong_count = 0

    for token_str in vocab:
        indices = get_last_syllable_jamo(token_str)
        if indices is not None:
            jamo_data.append(indices)
            if indices[2] == 0:
                no_jongseong_count += 1

    watermarkable = len(jamo_data)
    coverage = watermarkable / vocab_size

    choseong_dist = Counter(x for x, _, _ in jamo_data)
    jungseong_dist = Counter(y for _, y, _ in jamo_data)
    jongseong_dist = Counter(z for _, _, z in jamo_data)

    hash_uniformity: dict[str, dict] = {}
    for k_bits in [1, 2, 3]:
        policy = HashPolicy(mode="robustness", k_bits=k_bits)
        buckets = 2 ** k_bits
        ch_count = Counter(policy.calculate_channel_hashes(x, y, z)[0] for x, y, z in jamo_data)
        ju_count = Counter(policy.calculate_channel_hashes(x, y, z)[1] for x, y, z in jamo_data)
        jo_count = Counter(policy.calculate_channel_hashes(x, y, z)[2] for x, y, z in jamo_data)

        def uniformity_score(counter: Counter, n: int, k: int) -> float:
            expected = n / k
            if expected == 0:
                return 0.0
            return max(abs(counter.get(b, 0) - expected) / expected for b in range(k))

        def starved_buckets(counter: Counter, n: int, k: int) -> list[int]:
            expected = n / k
            if expected == 0:
                return []
            return [b for b in range(k)
                    if counter.get(b, 0) < STARVED_THRESHOLD * expected]

        hash_uniformity[f"k_bits={k_bits}"] = {
            "buckets": buckets,
            "choseong_max_deviation": round(uniformity_score(ch_count, watermarkable, buckets), 4),
            "jungseong_max_deviation": round(uniformity_score(ju_count, watermarkable, buckets), 4),
            "jongseong_max_deviation": round(uniformity_score(jo_count, watermarkable, buckets), 4),
            "choseong_starved_buckets": starved_buckets(ch_count, watermarkable, buckets),
            "jungseong_starved_buckets": starved_buckets(ju_count, watermarkable, buckets),
            "jongseong_starved_buckets": starved_buckets(jo_count, watermarkable, buckets),
            "choseong_bucket_counts": dict(sorted(ch_count.items())),
            "jungseong_bucket_counts": dict(sorted(ju_count.items())),
            "jongseong_bucket_counts": dict(sorted(jo_count.items())),
        }

    balanced_luts = {
        f"k_bits={k_bits}": build_balanced_lut(jamo_data, k_bits)
        for k_bits in [1, 2, 3]
    }

    return {
        "model_name": model_name,
        "vocab_size": vocab_size,
        "watermarkable_tokens": watermarkable,
        "coverage_pct": round(coverage * 100, 2),
        "no_jongseong_pct": round(no_jongseong_count / watermarkable * 100, 2) if watermarkable else 0,
        "has_jongseong_pct": round((watermarkable - no_jongseong_count) / watermarkable * 100, 2) if watermarkable else 0,
        "choseong_unique_values": len(choseong_dist),
        "jungseong_unique_values": len(jungseong_dist),
        "jongseong_unique_values": len(jongseong_dist),
        "starved_threshold": STARVED_THRESHOLD,
        "hash_uniformity": hash_uniformity,
        "balanced_luts": balanced_luts,
    }


def print_summary(r: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Vocabulary Analysis: {r['model_name']}")
    print("=" * 60)
    print(f"  Vocab size               : {r['vocab_size']:,}")
    print(f"  Watermarkable tokens     : {r['watermarkable_tokens']:,}  ({r['coverage_pct']}%)")
    print(f"  Tokens with Jongseong    : {r['has_jongseong_pct']}%")
    print(f"  Tokens without Jongseong : {r['no_jongseong_pct']}%")
    print(f"  Unique Choseong values   : {r['choseong_unique_values']} / 19")
    print(f"  Unique Jungseong values  : {r['jungseong_unique_values']} / 21")
    print(f"  Unique Jongseong values  : {r['jongseong_unique_values']} / 28")

    print("\n  Hash uniformity (max bucket deviation from uniform):")
    print(f"  {'k_bits':<10} {'Choseong':>12} {'Jungseong':>12} {'Jongseong':>12}")
    print("  " + "-" * 48)
    for key, val in r["hash_uniformity"].items():
        print(f"  {key:<10} {val['choseong_max_deviation']:>11.1%} "
              f"{val['jungseong_max_deviation']:>12.1%} "
              f"{val['jongseong_max_deviation']:>12.1%}")

    print(f"\n  Starved buckets (count < {int(STARVED_THRESHOLD*100)}% of uniform expectation):")
    for key, val in r["hash_uniformity"].items():
        ch = val["choseong_starved_buckets"]
        ju = val["jungseong_starved_buckets"]
        jo = val["jongseong_starved_buckets"]
        print(f"  {key:<10}  Choseong: {ch}  Jungseong: {ju}  Jongseong: {jo}")
    print("=" * 60)


def plot_heatmap(r: dict, out_path: str) -> None:
    """One subplot per k_bits value: a 3 × 2^k_bits heatmap of bucket counts."""
    keys = list(r["hash_uniformity"].keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4))
    if len(keys) == 1:
        axes = [axes]

    watermarkable = r["watermarkable_tokens"]
    channels = ("choseong", "jungseong", "jongseong")

    for ax, key in zip(axes, keys):
        val = r["hash_uniformity"][key]
        buckets = val["buckets"]
        matrix = np.zeros((3, buckets), dtype=float)
        for ch_idx, ch_name in enumerate(channels):
            counts = val[f"{ch_name}_bucket_counts"]
            for b in range(buckets):
                matrix[ch_idx, b] = counts.get(b, 0)

        # Normalize each row by uniform expectation so colors compare across channels.
        expected = watermarkable / buckets if buckets else 1.0
        normed = matrix / expected if expected else matrix

        im = ax.imshow(normed, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
        ax.set_xticks(range(buckets))
        ax.set_xticklabels([str(b) for b in range(buckets)])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Choseong", "Jungseong", "Jongseong"])
        ax.set_title(key)
        ax.set_xlabel("Hash bucket")
        for i in range(3):
            for j in range(buckets):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.suptitle("Tokens per (channel, bucket).  Color = ratio to uniform expectation.",
                 fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="× expected")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_lut_balance(results: dict, k_bits: int) -> None:
    key = f"k_bits={k_bits}"
    info = results["balanced_luts"][key]
    print(f"\n  Balanced LUT (k_bits={k_bits}, buckets={info['buckets']}) bucket sizes:")
    for ch_name, sizes in info["bucket_sizes"].items():
        max_dev = max(abs(s - sum(sizes) / len(sizes)) / max(sum(sizes) / len(sizes), 1)
                      for s in sizes)
        print(f"    {ch_name:<10}: {sizes}  (max_dev={max_dev:.1%})")


def run_analysis(model_name: str = "skt/kogpt2-base-v2",
                 lut_k_bits: int = 2) -> None:
    """
    Args:
        lut_k_bits: which k_bits' LUT to persist to hash_lut.json. Other
            k_bits values are still saved inside corpus_analysis.json under
            `balanced_luts`.
    """
    results = analyze_vocabulary(model_name)
    print_summary(results)
    _print_lut_balance(results, lut_k_bits)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "corpus_analysis.json")
    fig_path = os.path.join(RESULTS_DIR, "corpus_analysis_heatmap.png")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_heatmap(results, fig_path)

    # Persist the runtime-loadable LUT (single k_bits value) separately.
    lut_payload = {
        "model_name": model_name,
        "k_bits": lut_k_bits,
        **results["balanced_luts"][f"k_bits={lut_k_bits}"],
    }
    with open(HASH_LUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lut_payload, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved → {out_path}")
    print(f"Heatmap      → {fig_path}")
    print(f"Hash LUT     → {HASH_LUT_PATH}")


if __name__ == "__main__":
    run_analysis()
