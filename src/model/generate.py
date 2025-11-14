import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..watermark.processor import JamoWatermarkProcessor

def generate_watermarked_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    processor: JamoWatermarkProcessor,
    prompt: str,
    max_length: int = 50
) -> tuple[str, torch.LongTensor]:
    """
    Generates watermarked text using the provided model, tokenizer, and processor.

    Returns:
        A tuple containing the decoded watermarked text and the raw output tensor.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        logits_processor=[processor],
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id
    )
    watermarked_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return watermarked_text, outputs