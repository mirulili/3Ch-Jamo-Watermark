# --- Constants Definition ---

# Unicode start point for Hangul Syllables (가)
HANGUL_START_CODE = 0xAC00
# Unicode end point for Hangul Syllables (힣)
HANGUL_END_CODE = 0xD7A3

# (Number of Jungseong 21 * Number of Jongseong 28)
JUNGSEONG_X_JONGSEONG_COUNT = 588
# Number of Jongseong 28
JONGSEONG_COUNT = 28

def get_last_syllable_jamo(token_str: str) -> tuple[int, int, int] | None:
    """
    Decomposes only the last syllable of a token string into Choseong/Jungseong/Jongseong.
    Returns the integer indices of the decomposed Jamo.
    Returns None if no Hangul syllable is found in the token string.
    """
    last_syllable = None
    
    # 1. Iterate through the token string from the end to find the 'last syllable'. (e.g., '요' in '안녕하세요')
    for char in reversed(token_str):
        char_code = ord(char)
        # Check if it's within the Hangul Syllable range '가'(AC00) ~ '힣'(D7A3)
        if HANGUL_START_CODE <= char_code <= HANGUL_END_CODE:
            last_syllable = char
            break
            
    # 2. If the token has no last syllable (e.g., '.', '<s>')
    if last_syllable is None:
        # Return None to indicate that this token cannot be used for Jamo watermarking.
        return None

    # 3. Calculate Jamo indices using Unicode composition principles
    relative_code = ord(last_syllable) - HANGUL_START_CODE
    
    # Choseong index (x)
    choseong_index = relative_code // JUNGSEONG_X_JONGSEONG_COUNT
    
    # Remaining code after excluding Choseong
    remaining_code = relative_code % JUNGSEONG_X_JONGSEONG_COUNT
    
    # Jungseong index (y)
    jungseong_index = remaining_code // JONGSEONG_COUNT
    
    # Jongseong index (z)
    jongseong_index = remaining_code % JONGSEONG_COUNT
    
    # Return (x, y, z) integer tuple
    return (choseong_index, jungseong_index, jongseong_index)

# --- (Reference) Test Code ---
#if __name__ == "__main__":
#    token1 = "안녕하세요" # Last syllable: '요'
#    token2 = "워터마크"   # Last syllable: '크'
#    token3 = "BPE."      # No last syllable
#
#    # '요' = 'ㅇ' + 'ㅛ' + ''
#    # Indices: 11, 12, 0
#    print(f"'{token1}' -> {get_last_syllable_jamo(token1)}")
#    
#    # '크' = 'ㅋ' + 'ㅡ' + ''
#    # Indices: 15, 18, 0
#    print(f"'{token2}' -> {get_last_syllable_jamo(token2)}")
#
#    # No Hangul syllable -> default (0, 0, 0)
#    print(f"'{token3}' -> {get_last_syllable_jamo(token3)}")