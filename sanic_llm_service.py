from sanic import Sanic, response
from sanic.request import Request
from sanic_ext import Extend
import aiohttp
import json
import logging

from preprocessor import PreProcessor
from postprocessor import PostProcessor
from client_adapter import HTTPModelClientAdapter
from data_model.data_models import LLMGenContext, ClientResponse
from config import Config

app = Sanic("LLMService")
Extend(app)

# Set up logging
logger = logging.getLogger("LLMService")
logging.basicConfig(level=logging.INFO)

# Initialize components
adapter = HTTPModelClientAdapter()
preprocessor = PreProcessor(adapter=adapter)
postprocessor = PostProcessor()

# Configurable endpoint and auth key
MODEL_ENDPOINT = Config.MODEL_ENDPOINT
AUTH_KEY = Config.AUTH_KEY

@app.post("/v2/models/llm_generic_reasoner_test/infer")
async def infer(request: Request):
    try:
        payload = request.json
        inputs = payload.get("inputs", [])
        request_tensor = next((item["data"][0] for item in inputs if item["name"] == "request"), None)
        options_tensor = next((item["data"][0] for item in inputs if item["name"] == "options"), "{}")
        metadata_tensor = next((item["data"][0] for item in inputs if item["name"] == "request_metadata"), "{}")

        sampling_parameters = json.dumps({
            "request_metadata": json.loads(metadata_tensor),
            **json.loads(options_tensor)
        })

        request = {
            "inputs": {
                "text": request_tensor,
                "sampling_parameters": sampling_parameters,
                "stream": "false"
            }
        }

        context_map = preprocessor.get_contexts_from_request(request)
        context = list(context_map.values())[0]
        request_id_from_input = payload.get("id", context.request_id)

        if context.error_data:
            return response.json({"response": None, "error": context.error_data, "response_metadata": {}}, status=400)

        model_payload = {
            "model": context.request_parameters.get("model", "llm_generic_large"),
            "messages": [
                {"role": "user", "content": context.request_input.prompt}
            ],
            "temperature": context.request_parameters.get("temperature", 0.7),
            "top_p": context.request_parameters.get("top_p", 1.0),
            "max_tokens": context.request_parameters.get("max_tokens", 512),
            "stream": False,
            "request_metadata": {
                "trace_id": context.request_metadata.get("trace_id") or context.request_id
            }
        }

        headers = {
            "Authorization": f"Bearer {AUTH_KEY}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(MODEL_ENDPOINT, json=model_payload, headers=headers, ssl=False) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Model call failed: {error}")
                    return response.json({
                        "response": None,
                        "error": "Model inference failed",
                        "response_metadata": {}
                    }, status=500)

                model_response = await resp.json()

        # Extract from OpenAI-style response
        model_output = model_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        llm_response = ClientResponse(
            request_id=context.request_id,
            completion=model_output,
            error=None,
            stats=model_response.get("usage", {})
        )

        final_output = postprocessor.format_native_response(response=llm_response, context=context,
                                                            request_id=request_id_from_input)
        return response.json(final_output)

    except Exception as e:
        logger.exception("Unhandled error during inference")
        return response.json({"response": None, "error": str(e), "response_metadata": {}}, status=500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, access_log=True)