class HashPolicy:
    """
    Defines the policy for calculating Jamo-based hashes.
    This allows for easy swapping of different hashing strategies.
    """
    def __init__(self, mode: str = 'robustness', k_bits: int = 2):
        """
        Args:
            mode (str): 'robustness' or 'quality'.
            k_bits (int): The number of bits to embed, determining the hash range.
        """
        self.mode = mode
        self.k_bits = k_bits

    def calculate_channel_hashes(self, x: int, y: int, z: int) -> tuple[int, int, int]:
        """
        Calculates separate hash values for each Jamo channel.
        """
        mod_val = 2**self.k_bits
        
        choseong_hash = 0 if self.mode == 'quality' else x % mod_val
        jungseong_hash = y % mod_val
        jongseong_hash = z % mod_val

        return (choseong_hash, jungseong_hash, jongseong_hash)