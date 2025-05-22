import json
import time
from response_formatter import ResponseFormatter
from data_model.data_models import LLMGenContext, ClientResponse


class PostProcessor:

    def __init__(self):
        self.response_formatter = ResponseFormatter()

    def format_native_response(self, response: ClientResponse, context: LLMGenContext, request_id: str) -> dict:
        """
        Format response to include:
        - outputs[0]: model_output only
        - outputs[1]: everything else in response_metadata
        """
        start_time = time.time()

        print("context::", context)
        context.llm_response.raw_generated_text = response.completion
        context.post_processing_time = round((time.time() - start_time) * 1000, 2)

        if response.error:
            context.error_data = response.error
            model_output = None
        else:
            self.response_formatter.format_output(context)
            model_output = context.llm_response.formatted_text

        input_dict = json.loads(model_output)
        input_dict_model_output_inter_resp = json.loads(input_dict["response"])
        input_dict_model_output_resp = json.loads(input_dict_model_output_inter_resp['model_output'])
        input_dict_model_output_final_resp = input_dict_model_output_resp['response']
        input_dict_model_output_resp_metadata = json.loads(input_dict["response_metadata"])

        # Serialize only the model_output as first output
        response_block = {
            "name": "response",
            "datatype": "BYTES",
            "shape": [1],
            "data": [json.dumps({"model_output": input_dict_model_output_final_resp})]
        }

        # Build metadata by serializing everything else safely
        metadata = {
            "trace_id": context.request_metadata.get("trace_id", ""),
            "request_id": context.request_id,
            "request_parameters": context.request_parameters,
            "request_metadata": context.request_metadata,
            "preprocessing_time_ms": context.pre_processing_time,
            "postprocessing_time_ms": context.post_processing_time,
            "raw_model_output": response.completion,
            "llm_input": context.request_input.get_turns_for_prompt() if context.request_input else [],
            "stats": response.stats or {},
            "error": context.error_data
        }

        metadata_block = {
            "name": "response_metadata",
            "datatype": "BYTES",
            "shape": [],
            "data": [json.dumps(input_dict_model_output_resp_metadata, ensure_ascii=False)]
        }

        return {
            "id": request_id,
            "model_name": context.request_parameters.get("model_name", "llm_generic"),
            "model_version": context.request_parameters.get("model_version", "1"),
            "outputs": [response_block, metadata_block]
        }
