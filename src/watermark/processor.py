import torch
from transformers import LogitsProcessor
from .jamo_utils import get_last_syllable_jamo
from .hash_policy import HashPolicy

class JamoWatermarkProcessor(LogitsProcessor):
    """
    A watermark injector using the 3 channels of Korean Jamo (Choseong, Jungseong, Jongseong).
    Inherits from LogitsProcessor to intervene in the generate() pipeline in real-time.
    """
    def __init__(self, tokenizer, original_message: str, mode: str = 'robustness', k_bits: int = 2, top_k: int = 50, debug: bool = False):
        self.tokenizer = tokenizer    # Tokenizer for decoding
        byte_data = original_message.encode('utf-8')
        self.payload = ''.join(format(byte, '08b') for byte in byte_data)    # Full bitstream
        self.step_t = 0               # Current timestep (index of the next bit to insert)
        self.mode = mode              # 'robustness' or 'quality'
        self.debug = debug            # Debug mode flag
        self.k_bits = k_bits          # Number of bits to insert at once
        self.top_k = top_k            # Number of candidate tokens to consider
        self.hash_policy = HashPolicy(mode=self.mode, k_bits=self.k_bits)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        The main function called at each token generation step (watermark injection logic).
        """
        # 1. Get the target bits to insert in this step
        # (Extract k_bits from self.payload based on self.step_t)
        if self.step_t * self.k_bits >= len(self.payload):
            # Stop watermarking if payload is exhausted
            return scores
            
        target_bits = int(self.payload[self.step_t * self.k_bits : (self.step_t + 1) * self.k_bits], 2)

        # 1-2. Select a channel (0: Choseong, 1: Jungseong, 2: Jongseong) in a round-robin fashion
        channel_idx = self.step_t % 3

        # 2. Extract top K candidate tokens
        # (Get top_k from scores and decode to strings using the tokenizer)
        top_k_indices = scores.topk(self.top_k).indices[0]

        if self.debug:
            print(f"\n--- Step {self.step_t} ---")
            print(f"Channel: {['초성', '중성', '종성'][channel_idx]}, Target Bits: {target_bits} ('{self.payload[self.step_t * self.k_bits : (self.step_t + 1) * self.k_bits]}')")
            print("Top-K Candidates:")

        # 3. Bias the logits to steer generation towards tokens matching the target bits
        bias_value = 10.0  # A hyperparameter to control watermark strength
        for token_idx in top_k_indices: # Iterate through candidate tokens
            token_str = self.tokenizer.convert_ids_to_tokens(token_idx.item())
            jamo_indices = get_last_syllable_jamo(token_str)

            # Skip tokens that do not contain a Hangul syllable
            if jamo_indices is None:
                continue
            
            # Calculate hashes for the token
            token_hashes = self.hash_policy.calculate_channel_hashes(*jamo_indices)
            
            # Apply bias only if the hash of the selected channel matches the target bits
            if token_hashes[channel_idx] == target_bits:
                if self.debug:
                    print(f"  - ✅ Match: '{token_str}' (Hash: {token_hashes[channel_idx]}) -> Applying bias")
                # Add a bias to the logits of matching tokens
                scores[0, token_idx] += bias_value
            else:
                if self.debug:
                    print(f"  - ❌ No Match: '{token_str}' (Hash: {token_hashes[channel_idx]})")
        
        # 4. Check if a watermarked token is now the most likely candidate.
        # This is the core logic to synchronize the processor and detector.
        top_candidate_after_bias = scores.argmax()
        token_str = self.tokenizer.convert_ids_to_tokens(top_candidate_after_bias.item())
        jamo_indices = get_last_syllable_jamo(token_str)

        watermark_will_be_embedded = False
        if jamo_indices is not None:
            token_hashes = self.hash_policy.calculate_channel_hashes(*jamo_indices)
            if token_hashes[channel_idx] == target_bits:
                watermark_will_be_embedded = True

        if watermark_will_be_embedded:
            if self.debug:
                print(f"--> ✅ SUCCESS: Top candidate '{token_str}' matches target. Advancing step.")
            self.step_t += 1
        else:
            if self.debug:
                print(f"--> ❌ SKIP: Top candidate '{token_str}' does not match target. Step not advanced.")

        return scores