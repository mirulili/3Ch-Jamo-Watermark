import json
import os
from datetime import datetime

from .watermark.processor import JamoWatermarkProcessor
from .watermark.detector import JamoWatermarkDetector
from .watermark.payload_mgr import PayloadManager
from .model.load_model import load_model_and_tokenizer
from .model.generate import generate_watermarked_text
from .config import ExperimentConfig, set_seed, load_lut


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(config: ExperimentConfig | None = None):
    if config is None:
        config = ExperimentConfig()

    set_seed(config.seed)

    # --- 1. Configuration ---
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    lut = load_lut(config.lut_path)
    print(f"Hash LUT: {'loaded' if lut else 'NOT FOUND — falling back to modulo'}"
          f" ({config.lut_path})")

    original_message = "Read Me If You Can"

    # --- 2. Encoding (Watermark Generation) ---
    print("--- 1. Watermark Generation Phase ---")

    payload_mgr = PayloadManager()
    payload_bits = payload_mgr.encode(original_message)
    print(f"Original Message: '{original_message}'")

    jamo_processor = JamoWatermarkProcessor(
        tokenizer=tokenizer,
        mode=config.mode,
        k_bits=config.k_bits,
        top_k=config.top_k,
        bias_value=config.bias_value,
        lut=lut,
    )

    prompt = "인공지능은 인류의 삶을 어떻게 바꿀 것인가?"
    watermarked_text, outputs, prompt_length = generate_watermarked_text(
        model=model,
        tokenizer=tokenizer,
        processor=jamo_processor,
        prompt=prompt,
        payload=payload_bits,
        k_bits=config.k_bits,
        max_length=config.max_length,
    )

    print("\n[Generated Watermarked Text]")
    print(watermarked_text)
    print("-" * 30)

    # --- 3. Decoding (Watermark Extraction) ---
    print("\n--- 2. Watermark Detection Phase ---")

    detector = JamoWatermarkDetector(
        tokenizer=tokenizer,
        original_message=original_message,
        mode=config.mode,
        k_bits=config.k_bits,
        lut=lut,
    )

    result = detector.extract_payload(
        outputs, payload_bits, prompt_length=prompt_length
    )
    print(f"Extracted Payload (bits): '{result.extracted_payload}'")
    print(f"Recovery Rate: {result.accuracy * 100:.1f}%  ({min(result.detected_cnt, result.total_steps)}/{result.total_steps} bits of one cycle)")
    print(f"Cycles completed: {result.cycles_completed:.2f}  (detected_cnt={result.detected_cnt})")
    print(f"Z-Score:       {result.z_score:.2f}")
    print(f"Hangul trials: {result.n_hangul_trials}  |  max consecutive skip: {result.max_skip_run}")
    if result.channel_trials:
        per_ch = " ".join(
            f"ch{ch}={result.channel_matches.get(ch, 0)}/{result.channel_trials[ch]}"
            for ch in sorted(result.channel_trials)
        )
        print(f"Per-channel matches: {per_ch}")

    recovered_message = payload_mgr.decode(result.extracted_payload)
    print(f"Recovered Message: '{recovered_message}'")

    # --- 4. Verification ---
    print("\n--- 3. Verification ---")

    # Watermark presence is established by statistical significance, not by
    # recovery rate. A non-blind detector with greedy synchronization can
    # mechanically reach 100% recovery on unwatermarked text given enough
    # Hangul tokens, so recovery_rate alone is not evidence of a watermark.
    Z_THRESHOLD = 3.0
    if result.z_score >= Z_THRESHOLD:
        print(f"[Detection Success] Watermark Confirmed.  (z={result.z_score:.2f} ≥ {Z_THRESHOLD})")
    else:
        print(f"[Detection Fail] Watermark Not Confirmed.  (z={result.z_score:.2f} < {Z_THRESHOLD})")

    if recovered_message and original_message in recovered_message:
        print("[Recovery Success] Original message string was reconstructed from the bit stream.")
    else:
        print("[Recovery Fail] Original message could not be reconstructed.")

    # --- 5. Diagnostic Log ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"detection_log_main_{timestamp}.json")
    log_payload = {
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
        "original_message": original_message,
        "payload_bits": payload_bits,
        "watermarked_text": watermarked_text,
        "recovered_message": recovered_message,
        "z_threshold": Z_THRESHOLD,
        "detection": result.to_dict(),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=2)
    print(f"\nDetection log → {log_path}")


if __name__ == "__main__":
    main()
