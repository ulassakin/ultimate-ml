from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


# Update these defaults when OpenAI changes prices, or use a local settings override.
# They are estimates only; the OpenAI Platform remains the billing authority.
DEFAULT_MODEL = "gpt-5.4-mini"
MODEL_PRICING = {
    "gpt-5.4-mini": ModelPricing(0.75, 0.075, 4.50),
    "gpt-5.4-nano": ModelPricing(0.20, 0.02, 1.25),
}


def pricing_for(model: str, override: dict | None = None) -> ModelPricing:
    if override:
        return ModelPricing(**override)
    return MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])


def estimate_cost(model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
                  override: dict | None = None) -> float:
    prices = pricing_for(model, override)
    uncached = max(0, input_tokens - cached_input_tokens)
    return (uncached * prices.input_per_million + cached_input_tokens * prices.cached_input_per_million +
            output_tokens * prices.output_per_million) / 1_000_000

