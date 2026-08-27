"""Existing paid drafts are repaired only by the reviewer, with recoverable revisions."""
import json

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.provider import ProviderResult
from backend.ai.service import AIService


def legacy_payload():
    return {"title": "Legacy Representation Topic", "category": "ml_fundamentals", "difficulty": "intermediate",
        "one_sentence_summary": "A legacy paid draft.", "quick_recall": "Legacy recall.",
        "core_explanation": "The original paid explanation.", "related_topic_ids": ["cnn"]}


class ReviewerOnlyProvider:
    def __init__(self): self.calls=[]
    def structured(self, **kwargs):
        self.calls.append(kwargs["schema_name"])
        if kwargs["schema_name"] == "ultimate_ml_relationship_resolution":
            request=json.loads(kwargs["input_text"])
            assert "topic_catalog" not in request
            return ProviderResult({"prerequisites":[],"related":[],"rejected_candidates":[]}, 5, 5)
        assert kwargs["schema_name"] == "ultimate_ml_topic_quality_review"
        candidate=json.loads(kwargs["input_text"])["candidate"]
        assert candidate["core_explanation"] in {"The original paid explanation.", "The repaired explanation is more specific."}
        corrected={"title":"Legacy Representation Topic", "category":"representation_learning", "difficulty":"intermediate",
            "concept_type":"broad_concept", "one_sentence_summary":"A repaired legacy paid draft.", "quick_recall":"Repaired recall.",
            "core_explanation":"The repaired explanation is more specific.", "related_topic_ids":[], "prerequisite_topic_ids":[]}
        return ProviderResult({"corrected_topic":corrected,"quality_report":{"confidence":"high","blocking_issues_fixed":[{"area":"taxonomy","message":"Corrected the category."}]}}, 30, 40)


def setup_client(tmp_path, monkeypatch, provider):
    database=main.Database(tmp_path / "quality.db")
    original=legacy_payload()
    database.create_draft("legacy", "topic", original["title"], original, {"request_cost_usd":0.0123})
    database.create_draft("questions", "question", "Question draft", {"topic_id":"gradient-descent", "questions":[]})
    database.review("unchanged-question", "good")
    monkeypatch.setattr(main, "database", database)
    monkeypatch.setattr(main, "ai_service", AIService(database, provider, api_key="fake"))
    return TestClient(main.app), database, original


def test_legacy_review_is_opt_in_idempotent_and_never_generates_topics_or_questions(tmp_path, monkeypatch):
    provider=ReviewerOnlyProvider()
    client, database, original=setup_client(tmp_path, monkeypatch, provider)
    question_before=database.get_draft("questions")
    history_before=database.progress()
    listed=client.get("/api/ai/topic-drafts").json()["drafts"]
    assert listed[0]["quality_review"]["status"] == "not_run"
    estimate=client.get("/api/ai/drafts/legacy/quality-review-estimate").json()
    assert estimate["original_generation_will_not_repeat"] is True and estimate["original_generation_estimated_cost_usd"] == 0.0123

    reviewed=client.post("/api/ai/drafts/legacy/quality-review", json={})
    assert reviewed.status_code == 200 and reviewed.json()["reused"] is False
    payload=database.get_draft("legacy")["payload"]
    assert payload["category"] == "representation_learning"
    assert payload["quality_review_state"] == "reviewed"
    assert provider.calls == ["ultimate_ml_topic_quality_review"]
    revisions=client.get("/api/ai/drafts/legacy/quality-revisions").json()["revisions"]
    assert {item["revision_type"] for item in revisions} == {"pre_quality_review", "quality_review"}
    before_revision=next(item for item in revisions if item["revision_type"] == "pre_quality_review")
    before_payload=client.get("/api/ai/drafts/legacy/quality-revisions/"+before_revision["id"]).json()["payload"]
    assert {key:value for key,value in before_payload.items() if key!="quality_review_state"} == original
    assert before_payload["quality_review_state"] == "not_run"

    reopened=client.post("/api/ai/drafts/legacy/quality-review", json={}).json()
    assert reopened["reused"] is True and provider.calls == ["ultimate_ml_topic_quality_review"]
    forced=client.post("/api/ai/drafts/legacy/quality-review", json={"force":True})
    assert forced.status_code == 200 and provider.calls == ["ultimate_ml_topic_quality_review", "ultimate_ml_topic_quality_review"]
    assert database.get_draft("questions") == question_before and database.progress() == history_before

    restored=client.post("/api/ai/drafts/legacy/quality-revisions/"+before_revision["id"]+"/restore").json()
    restored_payload=restored["payload"]
    assert {key:value for key,value in restored_payload.items() if key!="quality_review_state"} == original
    assert restored_payload["quality_review_state"] == "not_run"
    assert database.get_draft("legacy")["payload"] == restored_payload


def test_failed_existing_review_keeps_original_recoverable_and_never_calls_generator(tmp_path, monkeypatch):
    class FailingReviewer:
        def __init__(self): self.calls=[]
        def structured(self, **kwargs): self.calls.append(kwargs["schema_name"]); raise RuntimeError("provider unavailable")
    provider=FailingReviewer()
    client, database, original=setup_client(tmp_path, monkeypatch, provider)
    result=client.post("/api/ai/drafts/legacy/quality-review", json={})
    assert result.status_code == 502 and provider.calls == ["ultimate_ml_topic_quality_review"]
    assert database.get_draft("legacy")["payload"]["core_explanation"] == original["core_explanation"]
    revisions=database.list_draft_quality_revisions("legacy")
    assert revisions[0]["revision_type"] == "pre_quality_review" and revisions[0]["payload"] == original
    assert database.get_draft("legacy")["payload"]["quality_review_state"] == "failed"
