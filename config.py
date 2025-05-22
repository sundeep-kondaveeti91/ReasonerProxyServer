import os

class Config:
    MODEL_ENDPOINT = os.getenv("LLM_MODEL_ENDPOINT", "https://42865dc7-b356-4219-b074-d615d74b3314.job.console.elementai.com/v1/chat/completions")
    AUTH_KEY = os.getenv("LLM_AUTH_KEY", "8o30OElfDYV_D6YbbznT0A:GDC2BsXIfSdfjv9iWka3V4MkazpvHfe0cCwXohzbP0Q")
    TOKENIZER_MODEL_NAME = os.getenv("TOKENIZER_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
