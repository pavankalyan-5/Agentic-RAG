"""DeepEval's judge LLM, routed through our gateway so grading is cached, logged, and budget-capped.

DeepEval normally calls OpenAI itself. A DeepEvalBaseLLM subclass replaces that: DeepEval's metrics call
generate(prompt, schema=SomeModel) and expect an instance of that pydantic model back (or text when no
schema is given).
"""

import re

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel, ValidationError

from copilot.llm.gateway import LLMGateway, LLMOutputError, RequestRejected

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```|(\{.*\})", re.DOTALL)


class GatewayJudge(DeepEvalBaseLLM):
    def __init__(self, gateway: LLMGateway, model: str):
        self.gateway = gateway
        self.model_id = model
        super().__init__(model)

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return self.model_id

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_):
        messages = [{"role": "user", "content": prompt}]
        if schema is None:
            return self.gateway.complete(messages, model=self.model_id, operation="judge")
        try:
            return self.gateway.chat(messages, model=self.model_id, schema=schema, operation="judge")
        except (LLMOutputError, RequestRejected):
            # Strict structured outputs refused this schema/prompt: ask for plain JSON and validate it.
            text = self.gateway.complete(
                messages + [{"role": "user", "content": "Reply with only a JSON object matching this schema: "
                             f"{schema.model_json_schema()}"}],
                model=self.model_id, operation="judge",
            )
            return _parse_json(text, schema)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs):
        # Metrics run with async_mode=False; this exists because DeepEval requires it.
        return self.generate(prompt, schema=schema, **kwargs)


def _parse_json(text: str, schema: type[BaseModel]) -> BaseModel:
    match = _JSON_BLOCK.search(text)
    if not match:
        raise LLMOutputError(f"judge returned no JSON for {schema.__name__}")
    try:
        return schema.model_validate_json(match.group(1) or match.group(2))
    except ValidationError as exc:
        raise LLMOutputError(f"judge JSON didn't match {schema.__name__}: {exc}") from exc
