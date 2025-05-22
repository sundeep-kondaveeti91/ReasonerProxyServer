import logging
from transformers import AutoTokenizer

class HTTPModelClientAdapter:
    """
    Adapter that provides tokenizer and logger support for Sanic inference service.
    """

    def __init__(self, model_name_or_path="meta-llama/Llama-2-7b-chat-hf"):
        self.logger = logging.getLogger("HTTPModelClientAdapter")
        logging.basicConfig(level=logging.INFO)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.logger.info(f"LLaMA tokenizer loaded from: {model_name_or_path}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer