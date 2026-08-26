import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .openai_provider import OpenAIProvider
from .pricing import DEFAULT_MODEL, estimate_cost
from .provider import AIProvider, ProviderResult

logger = logging.getLogger("ultimate_ml.ai")


class AIUnavailableError(RuntimeError): pass
class BudgetExceededError(RuntimeError): pass
class StructuredOutputError(RuntimeError): pass


@dataclass
class AISettings:
    provider: str = "openai"
    model: str = DEFAULT_MODEL
    explanation_depth: str = "ultimate"
    enabled: bool = True
    monthly_budget_usd: float = 5.0
    pricing_override: dict | None = None


class AIService:
    def __init__(self, database, provider: AIProvider | None = None, api_key: str | None = None):
        self.database = database
        self._provider = provider
        self._api_key = api_key

    def settings(self) -> AISettings:
        stored = self.database.get_setting("ai_settings") or {}
        try:
            environment_budget = float(os.getenv("ULTIMATE_ML_MONTHLY_AI_BUDGET_USD", "5.00"))
        except ValueError:
            environment_budget = 5.0
        defaults = {"model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL), "monthly_budget_usd": environment_budget}
        return AISettings(**(defaults | stored))

    def update_settings(self, values: dict) -> AISettings:
        current = self.settings().__dict__
        allowed = set(AISettings.__annotations__)
        current.update({key: value for key, value in values.items() if key in allowed})
        settings = AISettings(**current)
        self.database.set_setting("ai_settings", settings.__dict__)
        return settings

    def key_configured(self) -> bool:
        return bool(self._api_key or os.getenv("OPENAI_API_KEY"))

    def public_settings(self) -> dict:
        settings = self.settings()
        return {**settings.__dict__, "api_key_configured": self.key_configured(),
                "connectivity": self.database.get_setting("ai_connectivity") or {
                    "key_detected": self.key_configured(), "key_valid": None,
                    "model_access": None, "billing_model_access": None, "checked_at": None}}

    def estimate_maximum(self, operation_type: str, settings: AISettings | None = None) -> float:
        settings = settings or self.settings()
        caps = {"topic_draft": (4000, 5000), "question_draft": (2600, 2600), "regenerate_section": (2200, 1800),
                "connectivity_test": (100, 32), "youtube_concept_extraction": (5000, 1600),
                "youtube_topic_expansion": (5000, 5000), "youtube_question_generation": (3000, 2600),
                "topic_quality_review": (6000, 5000), "youtube_topic_quality_review": (6500, 5000),
                "topic_quality_review_existing": (6500, 5000),
                "metadata_relationship_resolution": (2600, 1600)}
        input_tokens, output_tokens = caps[operation_type]
        return estimate_cost(settings.model, input_tokens, output_tokens, override=settings.pricing_override)

    def estimate_operations(self, operation_types: list[str], settings: AISettings | None = None) -> dict:
        settings = settings or self.settings()
        operations = [{"operation_type": item, "maximum_estimated_cost_usd": self.estimate_maximum(item, settings)} for item in operation_types]
        return {"operations": operations, "maximum_estimated_cost_usd": sum(item["maximum_estimated_cost_usd"] for item in operations),
                "remaining_budget_usd": self.usage_summary()["remaining_budget_usd"]}

    def require_budget_for_operations(self, operation_types: list[str]) -> dict:
        estimate = self.estimate_operations(operation_types)
        if estimate["maximum_estimated_cost_usd"] > estimate["remaining_budget_usd"]:
            raise BudgetExceededError("Monthly local AI budget would be exceeded by generation plus quality review. Raise it in Settings only if you intentionally want to resume.")
        return estimate

    def usage_summary(self, calendar_month: str | None = None) -> dict:
        from datetime import datetime, timezone
        calendar_month = calendar_month or datetime.now(timezone.utc).strftime("%Y-%m")
        total, groups = self.database.monthly_usage(calendar_month)
        settings = self.settings()
        return {"calendar_month": calendar_month, "estimated_cost_usd": total, "monthly_budget_usd": settings.monthly_budget_usd,
                "remaining_budget_usd": max(0, settings.monthly_budget_usd - total), "groups": groups,
                "label": "Local estimate only; OpenAI Platform billing is authoritative."}

    def generate(self, *, operation_type, instructions, input_text, schema_name, schema, validate, max_output_tokens, metadata=None):
        settings = self.settings()
        if not settings.enabled:
            raise AIUnavailableError("AI authoring is disabled in Settings.")
        if not self.key_configured() and self._provider is None:
            raise AIUnavailableError("AI authoring needs a local OPENAI_API_KEY. Learning and review remain available without it.")
        projected = self.estimate_maximum(operation_type, settings)
        summary = self.usage_summary()
        if summary["estimated_cost_usd"] + projected > settings.monthly_budget_usd:
            raise BudgetExceededError("Monthly local AI budget would be exceeded. Raise it in Settings only if you intentionally want to resume.")
        event_id = self.database.create_usage_event(provider=settings.provider, model=settings.model, operation_type=operation_type,
            estimated_cost_usd=projected, metadata={"reserved_maximum_usd": projected, **(metadata or {})})
        try:
            provider = self._provider or OpenAIProvider(self._api_key or os.environ["OPENAI_API_KEY"])
            result = provider.structured(model=settings.model, instructions=instructions, input_text=input_text,
                schema_name=schema_name, schema=schema, max_output_tokens=max_output_tokens)
            try:
                validated = validate(result.payload)
            except Exception as exc:
                raise StructuredOutputError("The AI returned an invalid structured draft. No content was saved.") from exc
            actual = estimate_cost(settings.model, result.input_tokens, result.output_tokens, result.cached_input_tokens, settings.pricing_override)
            self.database.finalize_usage_event(event_id, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                cached_input_tokens=result.cached_input_tokens, estimated_cost_usd=actual)
            return validated, result, actual
        except Exception as exc:
            if not isinstance(exc, StructuredOutputError):
                error_type = type(exc).__name__
            else:
                error_type = "structured_output"
            self.database.finalize_usage_event(event_id, status="failure", estimated_cost_usd=0, error_type=error_type)
            self._log_provider_failure(operation_type, settings.model, exc)
            raise

    def connectivity_test(self) -> dict:
        settings = self.settings()
        result = {"key_detected": self.key_configured(), "key_valid": None, "model_access": None,
                  "billing_model_access": None, "checked_at": None}
        if not self.key_configured():
            self.database.set_setting("ai_connectivity", result)
            return result
        projected = self.estimate_maximum("connectivity_test", settings)
        summary = self.usage_summary()
        if summary["estimated_cost_usd"] + projected > settings.monthly_budget_usd:
            result["reason"] = "local_budget_blocked"
            self.database.set_setting("ai_connectivity", result)
            return result
        event_id = self.database.create_usage_event(provider=settings.provider, model=settings.model,
            operation_type="connectivity_test", estimated_cost_usd=projected,
            metadata={"reserved_maximum_usd": projected})
        try:
            provider = self._provider or OpenAIProvider(self._api_key or os.environ["OPENAI_API_KEY"])
            provider_result = provider.connectivity(model=settings.model)
            actual = estimate_cost(settings.model, provider_result.input_tokens, provider_result.output_tokens,
                provider_result.cached_input_tokens, settings.pricing_override)
            self.database.finalize_usage_event(event_id, input_tokens=provider_result.input_tokens,
                output_tokens=provider_result.output_tokens, cached_input_tokens=provider_result.cached_input_tokens,
                estimated_cost_usd=actual)
            result.update({"key_valid": True, "model_access": True, "billing_model_access": True,
                "checked_at": datetime.now(timezone.utc).isoformat()})
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            result.update({"key_valid": False if status == 401 else None, "model_access": False if status in (403, 404) else None,
                "billing_model_access": False if status in (402, 403, 429) else None,
                "checked_at": datetime.now(timezone.utc).isoformat()})
            self.database.finalize_usage_event(event_id, status="failure", estimated_cost_usd=0, error_type=type(exc).__name__)
            self._log_provider_failure("connectivity_test", settings.model, exc)
        self.database.set_setting("ai_connectivity", result)
        return result

    @staticmethod
    def _log_provider_failure(operation: str, model: str, exc: Exception) -> None:
        import re
        message = str(exc)
        message = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-[redacted]", message)
        message = re.sub(r"(?i)(authorization|api[_ -]?key)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", message)
        logger.warning("AI provider failure operation=%s model=%s exception_type=%s status=%s code=%s message=%s",
            operation, model, type(exc).__name__, getattr(exc, "status_code", None), getattr(exc, "code", None), message[:500])
