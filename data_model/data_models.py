from dataclasses import dataclass, field

@dataclass
class InputRequestData:
    prompt: str = None
    turns: list = field(default_factory=list)

    def get_turns_for_prompt(self):
        """
        Returns the turns as a list of dictionaries, compatible with JSON serialization.
        """
        return [{"role": t.role, "content": t.content} for t in self.turns]

@dataclass
class PreProcessedInput:
    original_input: InputRequestData
    preprocessed_input: InputRequestData

@dataclass
class LLMCompletionOutput:
    raw_generated_text: str = ""
    formatted_text: str = ""

@dataclass
class ClientResponse:
    request_id: str
    completion: str
    error: str = None
    stats: dict = field(default_factory=dict)

@dataclass
class LLMGenContext:
    request_id: str
    request_input: InputRequestData = None
    request_metadata: dict = field(default_factory=dict)
    request_parameters: dict = field(default_factory=dict)
    processed_prompt: InputRequestData = None
    llm_response: LLMCompletionOutput = field(default_factory=LLMCompletionOutput)
    error_data: str = None
    pre_processing_time: float = 0.0
    post_processing_time: float = 0.0
    streaming: bool = False
    tokenizer: object = None
    logger: object = None
