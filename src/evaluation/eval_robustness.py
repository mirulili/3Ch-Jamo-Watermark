import torch
from transformers import PreTrainedTokenizer

from ..watermark.detector import JamoWatermarkDetector
from ..watermark.payload_mgr import PayloadManager
from ..watermark.processor import JamoWatermarkProcessor
from ..model.load_model import load_model_and_tokenizer
from ..model.generate import generate_watermarked_text

def test_deletion_attack(
    watermarked_text: str,
    target_payload: str,
    tokenizer: PreTrainedTokenizer,
    detector: JamoWatermarkDetector
):

    print("\n" + "="*40)
    print("[Robustness Test: Deletion Attack]")
    print("="*40)

    # 1. 공격 시뮬레이션 (단어 삭제)
    words = watermarked_text.split(' ')
    if len(words) <= 10:
        print("Text is too short to perform deletion test.")
        return

    # 중간 지점 단어 하나 삭제 (예: n번째 단어)
    target_idx = 5
    removed_word = words[target_idx]
    del words[target_idx]
    attacked_text = ' '.join(words)
    
    print(f"Original Text Length: {len(watermarked_text)}")
    print(f"Attacked Text Length: {len(attacked_text)}")
    print(f"Removed Word: '{removed_word}'")
    
    # 2. 공격받은 텍스트 다시 토큰화
    attacked_input_ids = tokenizer.encode(attacked_text, return_tensors='pt')
    
    # 3. 검증 (Verification)
    # (공격으로 인해 비트가 밀렸을 것이나) 남은 부분들이 매칭되는지 확인
    accuracy, log, z_score = detector.extract_payload(attacked_input_ids, target_payload) 
    print(f"\n[Verification Result after Attack]")
    print(f"Log: {log}")
    print(f"Accuracy: {accuracy * 100:.1f}%")
    print(f"Z-Score : {z_score:.2f}")

    # 4. 판정
    # Z-Score 3.0 이상이면 우연일 확률이 0.1% 미만 -> 워터마크 확정
    # 또는 Accuracy가 랜덤(25%)보다 현저히 높은 50% 이상이면 성공으로 간주
    threshold = 0.50
    threshold_z = 3.0
    
    if accuracy >= threshold or z_score >= threshold_z:
        print(f"Success: Watermark survived deletion attack.")
    else:
        print(f"Fail: Watermark damaged too much.")

def run_test():

    print("--- Setting up Robustness Test ---")
    
    model_name = "skt/kogpt2-base-v2"
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    original_message = "ABC"
    mode = 'robustness'
    k_bits = 2
    
    payload_mgr = PayloadManager()
    payload_bits = payload_mgr.encode(original_message)
    print(f"Payload: {payload_bits}")

    print("\n--- Generating Watermarked Text ---")
    processor = JamoWatermarkProcessor(tokenizer, mode, k_bits, top_k=20)
    prompt = "인공지능은"
    
    watermarked_text, _ = generate_watermarked_text(
        model, tokenizer, processor, prompt, payload_bits, k_bits, max_length=200
    )
    print(f"Generated: {watermarked_text[:50]}...")

    detector = JamoWatermarkDetector(tokenizer, mode, k_bits)

    test_deletion_attack(watermarked_text, payload_bits, tokenizer, detector)

if __name__ == "__main__":
    run_test()