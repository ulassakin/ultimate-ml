"""Authoring and review are explicit one-call stages with no graph resolver."""
import json

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.provider import ProviderResult
from backend.ai.service import AIService


def authoring_payload():
    return {"title": "Fresh Topic", "category": "deep_learning", "difficulty": "intermediate", "concept_type": "broad_concept",
            "tags": ["learning"], "one_sentence_summary": "A concise summary.", "quick_recall": "A concise recall.",
            "core_explanation": "A technically correct explanation with practical ML relevance.",
            "prerequisite_topic_ids": ["cnn"], "related_topic_ids": ["resnet"], "metadata_resolution": {"legacy": True}}


def test_generate_and_explicit_review_each_make_one_provider_call(tmp_path, monkeypatch):
    class Provider:
        def __init__(self): self.calls = []
        def structured(self, **kwargs):
            self.calls.append(kwargs["schema_name"])
            if kwargs["schema_name"] == "ultimate_ml_topic_draft":
                return ProviderResult(authoring_payload(), 11, 12)
            candidate = json.loads(kwargs["input_text"])["candidate"]
            return ProviderResult({"corrected_topic": candidate, "quality_report": {"confidence": "high"}}, 13, 14)
    database, provider = main.Database(tmp_path / "app.db"), Provider()
    monkeypatch.setattr(main, "database", database)
    monkeypatch.setattr(main, "ai_service", AIService(database, provider, api_key="fake"))
    client = TestClient(main.app)
    created = client.post("/api/ai/topic-draft", json={"title": "Fresh Topic", "allow_duplicate": True})
    assert created.status_code == 200
    draft = created.json()
    assert provider.calls == ["ultimate_ml_topic_draft"]
    assert draft["payload"]["quality_review_state"] == "not_run"
    assert not any(field in draft["payload"] for field in ("prerequisite_topic_ids", "related_topic_ids", "metadata_resolution"))
    estimate = client.get(f"/api/ai/drafts/{draft['id']}/quality-review-estimate").json()
    assert [item["operation_type"] for item in estimate["operations"]] == ["topic_quality_review_existing"]
    reviewed = client.post(f"/api/ai/drafts/{draft['id']}/quality-review", json={})
    assert reviewed.status_code == 200
    assert provider.calls == ["ultimate_ml_topic_draft", "ultimate_ml_topic_quality_review"]
    assert reviewed.json()["draft"]["payload"]["quality_review_state"] == "reviewed"

