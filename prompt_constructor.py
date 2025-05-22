from string import Template
from data_model.data_models import InputRequestData, PreProcessedInput

class PromptConstructor:
    USER_TAG = "<|user|>"
    ASSISTANT_TAG = "<|assistant|>"
    END_TAG = "<|end|>"

    def _turns_to_prompt(self, turns):
        prompt = ""
        for turn in turns:
            role = turn.get("role")
            content = turn.get("content")
            tag = self.USER_TAG if role == "user" else self.ASSISTANT_TAG
            prompt += f"{tag}\n{content}\n{self.END_TAG}\n"
        prompt += self.ASSISTANT_TAG
        return prompt

    def build_prompt(self, context):
        request_data: InputRequestData = context.request_input
        if request_data.prompt:
            return PreProcessedInput(original_input=request_data, preprocessed_input=request_data)

        turns = request_data.turns
        if not turns:
            context.error_data = "Prompt or turns must be provided."
            return None

        prompt = self._turns_to_prompt(turns)
        processed_input = InputRequestData(prompt=prompt, turns=[])
        return PreProcessedInput(original_input=request_data, preprocessed_input=processed_input)
