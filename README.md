# ReasonerProxyServer

A lightweight, LLM inference server built using [Sanic](https://sanic.dev/). This framework handles:

- Chat-style prompt parsing  
- Tokenizer-based prompt construction (e.g. LLaMA)  
- Structured input parsing (Triton-style)  
- Clean OpenAI-style output formatting  
- ChatCompletions-compatible HTTP model calls  
- Native response formatting with optional JSON reasoning capture

---

## 🚀 Features

- ✅ Sanic-powered high-performance REST API  
- ✅ Plug-and-play tokenizer support (LLaMA via HuggingFace)  
- ✅ Support for structured input: `inputs`, `outputs`, and `id`  
- ✅ Parses `[BEGIN FINAL RESPONSE] ... [END FINAL RESPONSE]` block

## ⚙️ Requirements
```
pip install -r requirements.txt
```

## ▶️ Running the Server
```
python sanic_llm_service.py
```
