import torch
from transformers import PreTrainedTokenizer
from ..watermark.detector import JamoWatermarkDetector
from ..watermark.payload_mgr import PayloadManager

def test_deletion_attack(
    original_text: str,
    original_message: str,
    tokenizer: PreTrainedTokenizer,
    detector: JamoWatermarkDetector,
    payload_mgr: PayloadManager
):
    """
    Simulates a deletion attack on the watermarked text and checks for message recovery.
    """
    print("\n--- 4. Robustness Test (with text modification) ---")

    # Simulate a simple attack: remove a few words from the watermarked text
    words = original_text.split(' ')
    if len(words) <= 10:
        print("Text is too short to perform deletion test.")
        return

    # Remove the 5th word
    del words[5]
    modified_text = ' '.join(words)
    
    print(f"\n[Modified Text (removed a word)]\n{modified_text}")
    
    # Re-tokenize the modified text and try to extract the payload
    modified_input_ids = tokenizer.encode(modified_text, return_tensors='pt')
    
    extracted_payload = detector.extract_payload(modified_input_ids)
    recovered_message = payload_mgr.decode(extracted_payload)
    
    print(f"\nRecovered Message from modified text: '{recovered_message}'")
    if recovered_message and original_message in recovered_message:
        print("✅ Success: Message recovered even after text modification.")
    else:
        print("❌ Failure: Message could not be recovered after modification.")