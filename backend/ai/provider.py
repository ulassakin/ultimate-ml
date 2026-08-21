from dataclasses import dataclass
from typing import Protocol


@dataclass
class ProviderResult:
    payload: dict
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class AIProvider(Protocol):
    def structured(self, *, model: str, instructions: str, input_text: str, schema_name: str,
                   schema: dict, max_output_tokens: int) -> ProviderResult: ...

    def connectivity(self, *, model: str) -> ProviderResult: ...
