class HashPolicy:
    """
    Hash policy that maps a syllable's (Choseong, Jungseong, Jongseong) indices
    into per-channel bucket values and selects the active channel at each
    watermark step.

    Two bucketing modes:
    - Default (modulo): `jamo_index % 2^k_bits`. Fast but produces severely
      starved buckets for Korean (jongseong bucket 2/3 hold ~1.5% of tokens).
    - LUT-based: `lut[channel][jamo_index] → bucket`. Built from a frequency-
      balanced greedy partition of the vocabulary (see corpus_analysis.py).

    Channel selection is cyclic-aware: it takes (cycle_step, cycle_idx) and
    returns one of the 3 channels (or 2 in quality mode). The `+ cycle_idx`
    term guarantees every bit position visits every channel over successive
    cycles, regardless of whether `total_steps` shares a common factor with
    the channel count.
    """
    _CHANNEL_NAMES = ("choseong", "jungseong", "jongseong")

    def __init__(
        self,
        mode: str = 'robustness',
        k_bits: int = 2,
        lut: dict[str, dict[int, int]] | None = None,
    ):
        if mode not in ('robustness', 'quality'):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.k_bits = k_bits
        self.lut = lut  # None → use modulo fallback

    def calculate_channel_hashes(self, x: int, y: int, z: int) -> tuple[int, int, int]:
        """Per-channel hash bucket for one syllable."""
        if self.lut is not None:
            mod_val = 2 ** self.k_bits
            cho = self.lut.get('choseong', {}).get(x, x % mod_val)
            jung = self.lut.get('jungseong', {}).get(y, y % mod_val)
            jong = self.lut.get('jongseong', {}).get(z, z % mod_val)
            return (cho, jung, jong)
        mod_val = 2 ** self.k_bits
        return (x % mod_val, y % mod_val, z % mod_val)

    def get_channel_idx(self, cycle_step: int, cycle_idx: int = 0) -> int:
        """
        Channel index for the given (cycle_step, cycle_idx):
        - robustness: rotates over (Choseong, Jungseong, Jongseong)
        - quality:    rotates over (Jungseong, Jongseong) only
        """
        if self.mode == 'quality':
            return 1 + ((cycle_step + cycle_idx) % 2)
        return (cycle_step + cycle_idx) % 3
