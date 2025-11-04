from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str = "skt/kogpt2-base-v2"):
    """
    Loads a pre-trained model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer