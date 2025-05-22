import json
from data_model.data_models import LLMGenContext, InputRequestData
from prompt_constructor import PromptConstructor

class PreProcessor:
    def __init__(self, adapter):
        self._model_adapter = adapter

    def get_contexts_from_request(self, request):
        input_text = request["inputs"]["text"]
        sampling_parameters = request["inputs"]["sampling_parameters"]
        stream_input = request["inputs"].get("stream", "false") == "true"

        sampling_dict = json.loads(sampling_parameters)

        context = LLMGenContext(
            request_metadata=sampling_dict.get("request_metadata", {}),
            request_id=sampling_dict.get("id", "1")
        )
        context.request_parameters = sampling_dict

        # context = LLMGenContext(
        #     request_metadata=json.loads(json.loads(sampling_parameters).get("request_metadata", {})),
        #     request_id=json.loads(sampling_parameters).get("request_id", "generated-id")
        # )
        context.streaming = stream_input
        context.logger = self._model_adapter.logger

        input_data = InputRequestData(prompt=input_text, turns=[])
        context.request_input = input_data
        context.tokenizer = self._model_adapter.tokenizer

        prompt_constructor = PromptConstructor()
        context.processed_prompt = prompt_constructor.build_prompt(context)
        context.request_input = context.processed_prompt.preprocessed_input
        context.request_parameters = json.loads(sampling_parameters)

        return {context.request_id: context }
