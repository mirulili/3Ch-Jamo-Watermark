from .watermark.processor import JamoWatermarkProcessor
from .watermark.detector import JamoWatermarkDetector
from .watermark.payload_mgr import PayloadManager
from .evaluation.eval_robustness import test_deletion_attack
from .model.load_model import load_model_and_tokenizer
from .model.generate import generate_watermarked_text

def main():
    # --- 1. Configuration ---
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer("skt/kogpt2-base-v2")

    # Watermark parameters
    original_message = "ABC"
    mode = 'robustness'
    k_bits = 2

    # --- 2. Encoding (Watermark Generation) ---
    print("--- 1. Watermark Generation Phase ---")
    
    # PayloadManager is now used for safe encoding/decoding, even without BCH
    payload_mgr = PayloadManager()
    # The processor will handle the conversion of the message to bits internally
    print(f"Original Message: '{original_message}'")

    # Instantiate JamoWatermarkProcessor with the generated payload
    jamo_processor = JamoWatermarkProcessor(
        tokenizer=tokenizer,
        original_message = original_message,
        mode=mode,
        k_bits=k_bits,
        debug=True  # Enable debug logging
    )



    # Generate watermarked text using the new wrapper function
    prompt = "인공지능은 인류의 삶을 어떻게 바꿀 것인가?"
    watermarked_text, outputs = generate_watermarked_text(
        model=model,
        tokenizer=tokenizer,
        processor=jamo_processor,
        prompt=prompt
    )

    print("\n[Generated Watermarked Text]")
    print(watermarked_text)
    print("-" * 30)

    # --- 3. Decoding (Watermark Extraction) ---
    print("\n--- 2. Watermark Detection Phase ---")
    
    # Instantiate detector with the same settings
    detector = JamoWatermarkDetector(tokenizer=tokenizer, original_message=original_message, mode=mode, k_bits=k_bits)
    
    # Extract the bit payload from the generated text
    extracted_payload = detector.extract_payload(outputs)
    print(f"Extracted Payload (bits): '{extracted_payload}'")

    # Decode the extracted bits into a message using the payload manager
    recovered_message = payload_mgr.decode(extracted_payload)
    print(f"Recovered Message: '{recovered_message}'")
    
    # --- 4. Verification ---
    print("\n--- 3. Verification ---")
    if recovered_message and original_message in recovered_message:
        print("✅ Success: Original message was successfully recovered.")
    else:
        print("❌ Failure: Original message could not be recovered.")


if __name__ == "__main__":
    main()