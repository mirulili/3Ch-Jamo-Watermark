from .watermark.processor import JamoWatermarkProcessor
from .watermark.detector import JamoWatermarkDetector
from .watermark.payload_mgr import PayloadManager
from .model.load_model import load_model_and_tokenizer
from .model.generate import generate_watermarked_text

def main():


    # --- 1. Configuration ---
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer("skt/kogpt2-base-v2")

    # Watermark parameters
    original_message = "Read Me If You Can"
    mode = 'robustness'
    k_bits = 2


    # --- 2. Encoding (Watermark Generation) ---
    print("--- 1. Watermark Generation Phase ---")
    
    payload_mgr = PayloadManager()
    payload_bits = payload_mgr.encode(original_message)

    print(f"Original Message: '{original_message}'")

    # Instantiate JamoWatermarkProcessor with the generated payload
    jamo_processor = JamoWatermarkProcessor(
        tokenizer=tokenizer,
        mode=mode,
        k_bits=k_bits,
        top_k=20
    )

    # Generate watermarked text
    prompt = "인공지능은 인류의 삶을 어떻게 바꿀 것인가?"
    watermarked_text, outputs = generate_watermarked_text(
        model=model,
        tokenizer=tokenizer,
        processor=jamo_processor,
        prompt=prompt,
        payload=payload_bits,
        k_bits=k_bits,
    )

    print("\n[Generated Watermarked Text]")
    print(watermarked_text)
    print("-" * 30)


    # --- 3. Decoding (Watermark Extraction) --- 
    # Informed detection (verification)
    print("\n--- 2. Watermark Detection Phase ---")
    
    # Instantiate detector
    detector = JamoWatermarkDetector(tokenizer=tokenizer, original_message=original_message, mode=mode, k_bits=k_bits)
    
    # Extract the bit payload from the generated text
    accuracy, extracted_payload, z_score = detector.extract_payload(outputs, payload_bits)
    print(f"Extracted Payload (bits): '{extracted_payload}'")
    print(f"Accuracy: {accuracy * 100:.1f}%")
    print(f"Z-Score:  {z_score:.2f}")

    # Decode the extracted bits into a message using the payload manager
    recovered_message = payload_mgr.decode(extracted_payload)
    print(f"Recovered Message: '{recovered_message}'")
    
    # --- 4. Verification ---
    print("\n--- 3. Verification ---")

    if accuracy >= 0.95: 
        print("[Verification Success] Watermark Confirmed.")
    else:
        print("[Verification Fail] Watermark Not Found.")

    if recovered_message and original_message in recovered_message:
        print("[Recover Success] Original message was successfully verified.")
    else:
        print("[Recover Fail] Original message could not be verified.")


if __name__ == "__main__":
    main()