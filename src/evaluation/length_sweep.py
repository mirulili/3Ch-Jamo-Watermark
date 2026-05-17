"""
Message-length sweep: how does watermark detectability degrade as the payload
gets longer?

Hypothesis (from prior diagnostics): the per-Hangul z-score scales like
    z ≈ (signal_per_step · steps) / sqrt(total_trials)
where total_trials grows with text length and the bias mechanism produces
"stuck" mismatch streaks. As the message gets longer the generator must
produce more text, which inflates total_trials faster than detected_cnt grows
— so z-score collapses.

This script generates watermarked text for a battery of message lengths,
runs detection, and dumps both a per-length record and a length-vs-z-score
figure.
"""
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt

from ..config import ExperimentConfig, set_seed, load_lut
from ..model.generate import generate_watermarked_text
from ..model.load_model import load_model_and_tokenizer
from ..watermark.detector import JamoWatermarkDetector
from ..watermark.payload_mgr import PayloadManager
from ..watermark.processor import JamoWatermarkProcessor


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DEFAULT_PROMPT = "인공지능은 인류의 삶을 어떻게 바꿀 것인가?"
# An ASCII-only base so each character is exactly 1 byte (8 bits / 4 steps at k_bits=2).
DEFAULT_BASE_MESSAGE = "ABCDEFGHIJKLMNOP"
DEFAULT_LENGTHS = (1, 2, 4, 8, 16)


def _make_message(base: str, length: int) -> str:
    if length <= len(base):
        return base[:length]
    # Repeat the base if a length longer than it is requested.
    repeats = (length + len(base) - 1) // len(base)
    return (base * repeats)[:length]


def run_sweep(
    config: ExperimentConfig | None = None,
    prompt: str = DEFAULT_PROMPT,
    base_message: str = DEFAULT_BASE_MESSAGE,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
) -> dict:
    if config is None:
        config = ExperimentConfig()

    print(f"--- Length Sweep (seed={config.seed}, lengths={list(lengths)}) ---")
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    lut = load_lut(config.lut_path)
    print(f"Hash LUT: {'loaded' if lut else 'NOT FOUND — modulo fallback'}")
    payload_mgr = PayloadManager()

    print(
        f"\n{'Len':>4} {'Bits':>5} {'Steps':>6} {'Z':>8} "
        f"{'Recovery':>10} {'Cycles':>8} {'Trials':>8} {'MaxSkip':>8} {'GenTok':>8}"
    )
    print("-" * 82)

    records: list[dict] = []
    for length in lengths:
        # Each sweep point uses the same seed so that randomness in the model
        # / hash mismatch behavior is comparable across lengths.
        set_seed(config.seed)

        message = _make_message(base_message, length)
        payload_bits = payload_mgr.encode(message)

        processor = JamoWatermarkProcessor(
            tokenizer=tokenizer,
            mode=config.mode,
            k_bits=config.k_bits,
            top_k=config.top_k,
            bias_value=config.bias_value,
            lut=lut,
        )
        _, full_input_ids, prompt_length = generate_watermarked_text(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            prompt=prompt,
            payload=payload_bits,
            k_bits=config.k_bits,
            max_length=config.max_length,
        )

        detector = JamoWatermarkDetector(
            tokenizer=tokenizer,
            original_message=message,
            mode=config.mode,
            k_bits=config.k_bits,
            lut=lut,
        )
        result = detector.extract_payload(
            full_input_ids, payload_bits, prompt_length=prompt_length
        )
        generated_token_count = full_input_ids.shape[1] - prompt_length

        record = {
            "length_chars": length,
            "message": message,
            "payload_bits": len(payload_bits),
            "total_steps": result.total_steps,
            "detected_cnt": result.detected_cnt,
            "cycles_completed": round(result.cycles_completed, 4),
            "z_score": round(result.z_score, 4),
            "recovery_rate": round(result.accuracy, 4),
            "n_hangul_trials": result.n_hangul_trials,
            "max_skip_run": result.max_skip_run,
            "generated_tokens": generated_token_count,
            "channel_trials": result.channel_trials,
            "channel_matches": result.channel_matches,
        }
        records.append(record)

        print(
            f"{length:>4d} {len(payload_bits):>5d} {result.total_steps:>6d} "
            f"{result.z_score:>8.2f} {result.accuracy*100:>9.1f}% "
            f"{result.cycles_completed:>8.2f} "
            f"{result.n_hangul_trials:>8d} {result.max_skip_run:>8d} "
            f"{generated_token_count:>8d}"
        )

    return {
        "config": {
            "model_name": config.model_name,
            "mode": config.mode,
            "k_bits": config.k_bits,
            "top_k": config.top_k,
            "bias_value": config.bias_value,
            "max_length": config.max_length,
            "seed": config.seed,
            "lut_used": lut is not None,
            "lut_path": config.lut_path,
        },
        "prompt": prompt,
        "base_message": base_message,
        "records": records,
    }


def plot_sweep(records: list[dict], out_path: str) -> None:
    lengths = [r["length_chars"] for r in records]
    zs = [r["z_score"] for r in records]
    recoveries = [r["detected_cnt"] / r["total_steps"] if r["total_steps"] else 0
                  for r in records]
    trials = [r["n_hangul_trials"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_z = axes[0]
    ax_z.plot(lengths, zs, marker="o", color="#1f77b4", label="z-score")
    ax_z.axhline(3.0, color="gray", linestyle="--", linewidth=1, label="z = 3 (significance)")
    ax_z.set_xlabel("Message length (characters)")
    ax_z.set_ylabel("Z-score")
    ax_z.set_title("Detection significance vs. message length")
    ax_z.set_xscale("log", base=2)
    ax_z.set_xticks(lengths)
    ax_z.set_xticklabels([str(L) for L in lengths])
    ax_z.legend()
    ax_z.grid(True, linestyle=":", alpha=0.5)

    ax_r = axes[1]
    ax_r.plot(lengths, [r * 100 for r in recoveries], marker="s", color="#2ca02c",
              label="Recovery rate (%)")
    ax_r2 = ax_r.twinx()
    ax_r2.plot(lengths, trials, marker="^", color="#d62728", linestyle="--",
               label="Hangul trials (n)")
    ax_r.set_xlabel("Message length (characters)")
    ax_r.set_ylabel("Recovery rate (%)", color="#2ca02c")
    ax_r2.set_ylabel("Hangul trials (n)", color="#d62728")
    ax_r.set_title("Recovery rate and Hangul trial count")
    ax_r.set_xscale("log", base=2)
    ax_r.set_xticks(lengths)
    ax_r.set_xticklabels([str(L) for L in lengths])
    ax_r.set_ylim(0, 105)
    ax_r.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(config: ExperimentConfig | None = None) -> None:
    output = run_sweep(config)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"length_sweep_{timestamp}.json")
    fig_path = os.path.join(RESULTS_DIR, f"length_sweep_{timestamp}.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    plot_sweep(output["records"], fig_path)

    print(f"\nResults → {json_path}")
    print(f"Figure  → {fig_path}")


if __name__ == "__main__":
    run()
