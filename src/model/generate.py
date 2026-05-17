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
    max_length: int = 300,
) -> tuple[str, torch.LongTensor, int]:
    """
    Generates watermarked text using cyclic payload embedding: once step_t
    reaches total_steps the bit position wraps back to 0, so the payload is
    repeatedly embedded throughout the generated sequence rather than
    concentrated at the front.

    Channel selection depends on both `cycle_step` and `cycle_idx`, so each
    bit visits every channel over successive cycles even when `total_steps`
    happens to be a multiple of the channel count.

    Returns:
        (full_text, full_input_ids, prompt_length)
    """
    if len(payload) % k_bits != 0:
        raise ValueError(
            f"Payload length {len(payload)} is not a multiple of k_bits={k_bits}"
        )
    total_steps = len(payload) // k_bits
    if total_steps == 0:
        raise ValueError("Payload must encode at least one step")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    prompt_length = input_ids.shape[1]

    step_t = 0  # monotonic across cycles

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]

            cycle_step = step_t % total_steps
            cycle_idx = step_t // total_steps
            target_bits = int(
                payload[cycle_step * k_bits : (cycle_step + 1) * k_bits], 2
            )
            channel_idx = processor.hash_policy.get_channel_idx(cycle_step, cycle_idx)

            next_token_logits = processor.bias_logits(
                next_token_logits, target_bits, channel_idx
            )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if processor.check_token_match(next_token.item(), target_bits, channel_idx):
                step_t += 1
                print(f"Cycle {cycle_idx} Step {cycle_step} Success: "
                      f"{tokenizer.decode(next_token.item())}")
            else:
                print("Skip (Mismatch)")

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    watermarked_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return watermarked_text, input_ids, prompt_length
