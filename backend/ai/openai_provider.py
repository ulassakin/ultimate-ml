import json
from openai import OpenAI

from .provider import ProviderResult


class OpenAIProvider:
    """The only code path that imports and invokes the official OpenAI SDK."""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def structured(self, *, model, instructions, input_text, schema_name, schema, max_output_tokens):
        response = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=max_output_tokens,
            store=False,
            text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
        )
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None) if usage else None
        return ProviderResult(
            payload=json.loads(response.output_text),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
        )

    def connectivity(self, *, model: str) -> ProviderResult:
        # Retrieving the configured model checks authentication and model access
        # without generating content. The tiny response then verifies that a
        # billable Responses request can actually complete for this project.
        self.client.models.retrieve(model)
        response = self.client.responses.create(
            model=model, input="Reply with OK.", max_output_tokens=16, store=False,
            text={"verbosity": "low"},
        )
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None) if usage else None
        return ProviderResult(
            payload={"ok": True}, input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
        )
