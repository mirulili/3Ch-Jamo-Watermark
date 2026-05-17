"""
Robustness evaluation: tests watermark survival under deletion, substitution, and crop attacks.
Results are saved to results/robustness.json.
"""
import json
import random
import os

from ..watermark.detector import JamoWatermarkDetector
from ..watermark.payload_mgr import PayloadManager
from ..watermark.processor import JamoWatermarkProcessor
from ..model.load_model import load_model_and_tokenizer
from ..model.generate import generate_watermarked_text
from ..config import ExperimentConfig, set_seed, load_lut

ATTACK_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def _words(text: str) -> list[str]:
    return text.split()


def attack_deletion(text: str, rate: float, rng: random.Random) -> str:
    """Delete each word independently with probability `rate`."""
    words = _words(text)
    kept = [w for w in words if rng.random() >= rate]
    return " ".join(kept)


def attack_substitution(text: str, rate: float, rng: random.Random, vocab: list[str]) -> str:
    """Replace each word independently with probability `rate` with a random vocab token."""
    words = _words(text)
    result = [rng.choice(vocab) if rng.random() < rate else w for w in words]
    return " ".join(result)


def attack_crop(text: str, rate: float) -> str:
    """Crop the last `rate` fraction of words (keep the head)."""
    words = _words(text)
    keep_n = max(1, int(len(words) * (1.0 - rate)))
    return " ".join(words[:keep_n])


def attack_crop_from_front(text: str, rate: float) -> str:
    """Crop the first `rate` fraction of words (keep the tail).

    With non-cyclic embedding the watermark sits in the head and this attack
    is devastating. With cyclic embedding later cycles in the tail should
    keep the watermark detectable.
    """
    words = _words(text)
    drop_n = int(len(words) * rate)
    return " ".join(words[drop_n:]) if drop_n < len(words) else words[-1]


def attack_crop_middle(text: str, rate: float) -> str:
    """Remove a contiguous middle window of `rate` fraction of words."""
    words = _words(text)
    n = len(words)
    drop_n = int(n * rate)
    if drop_n <= 0 or drop_n >= n:
        return " ".join(words)
    start = (n - drop_n) // 2
    end = start + drop_n
    return " ".join(words[:start] + words[end:])


def evaluate_attack(attacked_text: str, tokenizer, detector: JamoWatermarkDetector,
                    target_payload: str) -> dict:
    # attacked_text contains only the generated portion (no prompt),
    # so detector reads from token 0 — prompt_length=0.
    input_ids = tokenizer.encode(attacked_text, return_tensors="pt")
    result = detector.extract_payload(input_ids, target_payload, prompt_length=0)
    return {
        "accuracy": round(result.accuracy, 4),
        "cycles_completed": round(result.cycles_completed, 4),
        "z_score": round(result.z_score, 4),
        "n_hangul_trials": result.n_hangul_trials,
        "max_skip_run": result.max_skip_run,
        "channel_trials": result.channel_trials,
        "channel_matches": result.channel_matches,
    }


def run_attack_suite(
    watermarked_text: str,
    target_payload: str,
    tokenizer,
    detector: JamoWatermarkDetector,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    vocab = list({t for t in tokenizer.get_vocab().keys() if any("가" <= c <= "힣" for c in t)})

    results = {}

    print("\n[Attack Suite]")
    # Accuracy is unreliable here (non-blind detector ceilings at 100%), so the
    # table prints z-score and diagnostic counters. Accuracy is kept in JSON.
    print(f"{'Attack':<14} {'Rate':>6} {'Z-Score':>10} {'Trials':>8} {'MaxSkip':>8} {'Pass':>6}")
    print("-" * 56)

    for attack_name, attack_fn in [
        ("deletion", lambda text, r: attack_deletion(text, r, rng)),
        ("substitution", lambda text, r: attack_substitution(text, r, rng, vocab)),
        ("crop_tail", lambda text, r: attack_crop(text, r)),
        ("crop_head", lambda text, r: attack_crop_from_front(text, r)),
        ("crop_middle", lambda text, r: attack_crop_middle(text, r)),
    ]:
        results[attack_name] = []
        for rate in ATTACK_RATES:
            attacked = attack_fn(watermarked_text, rate)
            metrics = evaluate_attack(attacked, tokenizer, detector, target_payload)
            passed = metrics["z_score"] >= 3.0
            results[attack_name].append({"rate": rate, **metrics, "passed": passed})
            print(f"{attack_name:<14} {rate:>6.0%} {metrics['z_score']:>10.2f} "
                  f"{metrics['n_hangul_trials']:>8d} {metrics['max_skip_run']:>8d} "
                  f"{'✓' if passed else '✗':>6}")

    return results


def run_test(config: ExperimentConfig | None = None):
    if config is None:
        config = ExperimentConfig()

    set_seed(config.seed)
    print(f"--- Robustness Evaluation (seed={config.seed}) ---")

    model, tokenizer = load_model_and_tokenizer(config.model_name)
    lut = load_lut(config.lut_path)
    print(f"Hash LUT: {'loaded' if lut else 'NOT FOUND — modulo fallback'}")

    original_message = "ABC"
    payload_mgr = PayloadManager()
    payload_bits = payload_mgr.encode(original_message)
    print(f"Message: '{original_message}'  Payload bits: {len(payload_bits)}")

    processor = JamoWatermarkProcessor(
        tokenizer, config.mode, config.k_bits, config.top_k, config.bias_value,
        lut=lut,
    )
    _, full_input_ids, prompt_length = generate_watermarked_text(
        model, tokenizer, processor,
        prompt="인공지능은",
        payload=payload_bits,
        k_bits=config.k_bits,
        max_length=config.max_length,
    )

    # Strip the prompt — attacks/detection should only see the watermarked portion.
    generated_text = tokenizer.decode(
        full_input_ids[0, prompt_length:], skip_special_tokens=True
    )
    print(f"\nGenerated ({len(generated_text)} chars): {generated_text[:80]}...")

    detector = JamoWatermarkDetector(
        tokenizer=tokenizer,
        original_message=original_message,
        mode=config.mode,
        k_bits=config.k_bits,
        lut=lut,
    )

    attack_results = run_attack_suite(generated_text, payload_bits, tokenizer, detector, config.seed)

    output = {
        "config": {
            "model_name": config.model_name,
            "mode": config.mode,
            "k_bits": config.k_bits,
            "bias_value": config.bias_value,
            "seed": config.seed,
            "message": original_message,
            "lut_used": lut is not None,
            "lut_path": config.lut_path,
        },
        "attacks": attack_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "robustness.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    run_test()
