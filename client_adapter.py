import logging
from transformers import AutoTokenizer
from config import Config


class HTTPModelClientAdapter:
    """
    Adapter that provides tokenizer and logger support for Sanic inference service.
    """

    def __init__(self, model_name_or_path="meta-llama/Llama-2-7b-chat-hf"):
        self.logger = logging.getLogger("HTTPModelClientAdapter")
        logging.basicConfig(level=logging.INFO)

        model_name = model_name_or_path or Config.TOKENIZER_MODEL_NAME

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info(f"Tokenizer loaded from: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer