import torch
from .jamo_utils import get_last_syllable_jamo
from .hash_policy import HashPolicy

class JamoWatermarkProcessor:
    """
    A watermark injector using the 3 channels of Korean Jamo (Choseong, Jungseong, Jongseong).
    Inherits from LogitsProcessor to intervene in the generate() pipeline in real-time.
    """
    def __init__(self, tokenizer, mode: str = 'robustness', k_bits: int = 2,
                 top_k: int = 30, bias_value: float = 5.0,
                 lut: dict[str, dict[int, int]] | None = None):
        self.tokenizer = tokenizer
        self.mode = mode
        self.k_bits = k_bits
        self.top_k = top_k
        self.bias_value = bias_value
        self.hash_policy = HashPolicy(mode=self.mode, k_bits=self.k_bits, lut=lut)

    def bias_logits(self, logits: torch.FloatTensor, target_bits: int, channel_idx: int) -> torch.FloatTensor:

        top_k_indices = logits.topk(self.top_k).indices[0]

        for token_idx in top_k_indices:
            token_str = self.tokenizer.convert_ids_to_tokens(token_idx.item())
            jamo_indices = get_last_syllable_jamo(token_str)

            # Skip tokens that do not contain a Hangul syllable
            if jamo_indices is None:
                continue
            
            # Calculate hashes for the token
            token_hashes = self.hash_policy.calculate_channel_hashes(*jamo_indices)
            
            # Apply bias only if the hash of the selected channel matches the target bits
            if token_hashes[channel_idx] == target_bits:
                logits[0, token_idx] += self.bias_value
        
        return logits
    
    def check_token_match(self, token_id: int, target_bits: int, channel_idx: int) -> bool:

        token_str = self.tokenizer.convert_ids_to_tokens(token_id)
        jamo_indices = get_last_syllable_jamo(token_str)

        if jamo_indices is None:
            return False

        token_hashes = self.hash_policy.calculate_channel_hashes(*jamo_indices)

        if token_hashes[channel_idx] == target_bits:
            return True
        
        return False