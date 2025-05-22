import re
import json
from common.constants import Constants

class ResponseFormatter:
    SPECIAL_TOKENS = ["</s>"]

    def _define_response_format(self, context):
        options = context.request_parameters or {}
        response_format = options.get(Constants.RESPONSE_FORMAT_PARAM_NAME, None)
        if not response_format:
            return None
        return response_format.get(Constants.FORMAT_TYPE, None)

    def _extract_json_part(self, text):
        matches = re.findall(r"[\[{].*?[\]}]", text, re.DOTALL)
        return matches[0] if matches else None

    def _parse_valid_json(self, text):
        try:
            return json.dumps(json.loads(text), ensure_ascii=False)
        except Exception:
            try:
                text = text.replace("\'", "'")
                return json.dumps(eval(text), ensure_ascii=False)
            except Exception:
                return None

    def _clean_output(self, text):
        for token in self.SPECIAL_TOKENS:
            text = text.replace(token, "")
        return text.strip()

    def _json_formatting(self, context):
        output_text = context.llm_response.raw_generated_text
        if Constants.PREFILL_ASSISTANT_CONTENT_OPTION in context.request_metadata:
            output_text = context.request_metadata[Constants.PREFILL_ASSISTANT_CONTENT_OPTION] + output_text

        output_text = self._clean_output(output_text)
        json_segment = self._extract_json_part(output_text)
        valid_json = self._parse_valid_json(json_segment or output_text)

        if valid_json:
            context.llm_response.formatted_text = valid_json
        else:
            context.error_data = "Invalid JSON returned by model."

    def format_output(self, context):
        response_format = self._define_response_format(context)
        context.llm_response.formatted_text = self._clean_output(
            context.llm_response.raw_generated_text
        )

        if not response_format:
            return

        if response_format in ["json", "json_object"] and not context.streaming:
            self._json_formatting(context)
            return

        if response_format not in Constants.SUPPORTED_FORMATS:
            context.error_data = f"Unknown response format: {response_format}"
