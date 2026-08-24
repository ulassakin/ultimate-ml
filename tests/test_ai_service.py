from datetime import datetime, timezone

import pytest

from backend.ai.provider import ProviderResult
from backend.ai.schemas import TopicDraft
from backend.ai.service import AIService, BudgetExceededError, StructuredOutputError
from backend.ai.structured import strict_response_schema
from backend.database import Database


VALID_TOPIC = {
    "title": "Gaussian Mixture Models", "category": "classical_ml", "difficulty": "intermediate", "one_sentence_summary": "A weighted sum of Gaussian components.",
    "quick_recall": "A GMM gives each point a soft component assignment.", "core_explanation": "It models data as a mixture.",
}


class FakeProvider:
    def __init__(self, payload=VALID_TOPIC): self.payload = payload
    def structured(self, **kwargs): return ProviderResult(self.payload, input_tokens=100, output_tokens=200)
    def connectivity(self, **kwargs): return ProviderResult({"ok": True}, input_tokens=4, output_tokens=2)


def test_fake_generation_logs_usage_without_real_api(tmp_path):
    db = Database(tmp_path / "ai.db")
    service = AIService(db, FakeProvider(), api_key="test-only-not-a-real-key")
    draft, result, cost = service.generate(operation_type="topic_draft", instructions="x", input_text="x", schema_name="x",
        schema=TopicDraft.model_json_schema(), validate=TopicDraft.model_validate, max_output_tokens=100)
    assert draft.title == "Gaussian Mixture Models"
    summary = service.usage_summary()
    assert summary["estimated_cost_usd"] == cost
    assert summary["groups"][0]["input_tokens"] == 100


def test_budget_is_a_preflight_hard_guard(tmp_path):
    db = Database(tmp_path / "ai.db")
    service = AIService(db, FakeProvider(), api_key="test")
    service.update_settings({"monthly_budget_usd": 0})
    with pytest.raises(BudgetExceededError):
        service.generate(operation_type="topic_draft", instructions="x", input_text="x", schema_name="x",
            schema=TopicDraft.model_json_schema(), validate=TopicDraft.model_validate, max_output_tokens=100)


def test_exactly_used_budget_blocks_the_next_request(tmp_path):
    db = Database(tmp_path / "ai.db")
    service = AIService(db, FakeProvider(), api_key="test")
    service.update_settings({"monthly_budget_usd": 5})
    db.create_usage_event(provider="openai", model="gpt-5.4-mini", operation_type="topic_draft", estimated_cost_usd=5, status="success")
    with pytest.raises(BudgetExceededError):
        service.generate(operation_type="topic_draft", instructions="x", input_text="x", schema_name="x",
            schema=TopicDraft.model_json_schema(), validate=TopicDraft.model_validate, max_output_tokens=100)


def test_invalid_structured_result_never_becomes_successful_usage(tmp_path):
    db = Database(tmp_path / "ai.db")
    service = AIService(db, FakeProvider({"title":"bad"}), api_key="test")
    with pytest.raises(StructuredOutputError):
        service.generate(operation_type="topic_draft", instructions="x", input_text="x", schema_name="x",
            schema=TopicDraft.model_json_schema(), validate=TopicDraft.model_validate, max_output_tokens=100)
    _, groups = db.monthly_usage(datetime.now(timezone.utc).strftime("%Y-%m"))
    assert groups[0]["estimated_cost_usd"] == 0


def test_monthly_usage_resets_by_calendar_month(tmp_path):
    db = Database(tmp_path / "ai.db")
    db.create_usage_event(provider="openai", model="gpt-5.4-mini", operation_type="topic_draft", estimated_cost_usd=1,
        status="success", now=datetime(2026, 7, 31, tzinfo=timezone.utc))
    assert db.monthly_usage("2026-08")[0] == 0


def test_public_settings_do_not_expose_a_key(tmp_path):
    service = AIService(Database(tmp_path / "ai.db"), api_key="secret-never-returned")
    settings = service.public_settings()
    assert settings["api_key_configured"] is True
    assert "api_key" not in settings


def test_environment_provides_initial_model_and_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("ULTIMATE_ML_MONTHLY_AI_BUDGET_USD", "2.50")
    service = AIService(Database(tmp_path / "ai.db"))
    assert service.settings().model == "gpt-5.4-nano"
    assert service.settings().monthly_budget_usd == 2.5


def test_drafts_persist_edits_and_state(tmp_path):
    db = Database(tmp_path / "ai.db")
    db.create_draft("draft-1", "topic", "Draft", {"title": "Draft"})
    assert db.get_draft("draft-1")["state"] == "draft"
    assert db.update_draft("draft-1", {"title": "Edited"}, "discarded")
    draft = db.get_draft("draft-1")
    assert draft["payload"]["title"] == "Edited"
    assert draft["state"] == "discarded"


def test_strict_schema_rejects_unknown_keys_and_requires_all_properties():
    schema = strict_response_schema(TopicDraft)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["$defs"]["EquationDraft"]["additionalProperties"] is False


def test_connectivity_check_records_key_model_and_billing_access(tmp_path):
    db = Database(tmp_path / "ai.db")
    service = AIService(db, FakeProvider(), api_key="test")
    result = service.connectivity_test()
    assert result["key_detected"] is True
    assert result["key_valid"] is True
    assert result["model_access"] is True
    assert result["billing_model_access"] is True
    assert db.get_setting("ai_connectivity")["billing_model_access"] is True


def test_provider_failure_logging_redacts_api_key(caplog):
    AIService._log_provider_failure("topic_draft", "gpt-5.4-mini", RuntimeError("Authorization: Bearer sk-secret-value"))
    assert "sk-secret-value" not in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
