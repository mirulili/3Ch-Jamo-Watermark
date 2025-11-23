import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..watermark.processor import JamoWatermarkProcessor

def generate_watermarked_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    processor: JamoWatermarkProcessor,
    prompt: str,
    payload: str,
    k_bits: int = 2,
    max_length: int = 300
) -> tuple[str, torch.LongTensor]:
    """
    Generates watermarked text using the provided model, tokenizer, and processor.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    step_t = 0

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]

            # Watermarking Logic
            # Check if there are remaining bits to embed
            if step_t * k_bits < len(payload):
                # Calculate current target bits and channel
                target_bits = int(payload[step_t * k_bits : (step_t + 1) * k_bits], 2)
                channel_idx = step_t % 3  # Cycle through 3 channels

                # 1) Biasing logits by calling Processor
                next_token_logits = processor.bias_logits(next_token_logits, target_bits, channel_idx)
            else:
                target_bits = None  # Watermaking done (no more bits to embed)

            # 2) Sample the next token
            # After softmax, sampling by multinomial
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  

            # 3) Check synchronization
            # Check if the chosen token satifies the watermark condition
            if target_bits is not None:
                is_match = processor.check_token_match(next_token.item(), target_bits, channel_idx)

                if is_match:
                    step_t += 1  # Move to the next set of bits only if matched
                    print(f"Step {step_t-1} Success: {tokenizer.decode(next_token.item())}")
                else:
                    print("Skip (Mismatch)")
                    pass
                    
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        
    watermarked_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return watermarked_text, input_ids