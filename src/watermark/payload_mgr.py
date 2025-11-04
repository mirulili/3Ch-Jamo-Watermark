class PayloadManager:
    """
    Manages the watermark payload, including message-to-bit conversion.
    """
    def __init__(self):
        """
        Initializes the PayloadManager.
        """
        pass

    def encode(self, message: str) -> str:
        """
        Encodes a string message into a bit string.
        
        Args:
            message (str): The message to encode.
            
        Returns:
            str: The payload as a binary string.
        """
        # Convert the string message to a byte array
        byte_data = message.encode('utf-8')
        payload_bits = ''.join(format(byte, '08b') for byte in byte_data)
        return payload_bits

    def decode(self, payload_bits: str) -> str | None:
        """
        Decodes a (potentially corrupted) bit string to the original message.
        """
        # Convert the bit string to a byte array
        try:
            # Ensure we only process full bytes (8 bits)
            num_bytes = len(payload_bits) // 8
            byte_data = bytearray(int(payload_bits[i*8:i*8+8], 2) for i in range(num_bytes))
            # Decode the byte array to a string, ignoring any decoding errors
            return byte_data.decode('utf-8', errors='ignore')
        except Exception:
            return None