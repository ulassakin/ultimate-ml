"""Regression fixtures for mistakes observed in paid topic generations; all calls are faked."""
import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.provider import ProviderResult
from backend.ai.schemas import TopicDraft
from backend.ai.service import AIService
from backend.ai.topic_quality import normalize_topic


CATALOG = [
    {"id": item, "title": item.replace("-", " ").title(), "category": "deep_learning", "summary": "Existing topic."}
    for item in ("gradient-descent", "cnn", "resnet", "gradient-flow", "self-supervised-learning", "covariance", "pca")
]


def topic(title, *, category="ml_fundamentals", relations=(), latex="z_s = x\""):
    return {"title": title, "category": category, "difficulty": "intermediate", "concept_type": "broad_concept",
        "tags": ["self-supervised learning", "representation learning"], "one_sentence_summary": "A concise summary.",
        "quick_recall": "A concise recall.", "core_explanation": "A technically bounded explanation.",
        "prerequisite_topic_ids": list(relations), "related_topic_ids": list(relations),
        "relationship_justifications": [{"topic_id": item, "relationship": "prerequisite", "reason": "This exists, so it is supposedly useful."} for item in relations],
        "mathematical_foundation": {"overview": "Math.", "sections": [{"title": "Objective", "explanation": "Defined symbols.", "equations": [{"latex": latex, "explanation": "A formula explanation."}]}]},
        "suggested_new_topic_relationships": [{"title": "Missing Neighbor", "relationship": "related", "reason": "A genuinely useful absent concept."}]}


def test_deterministic_regressions_remove_weak_edges_canonicalize_provenance_and_catch_bad_math():
    kd, _, blocking = normalize_topic(topic("Knowledge Distillation", relations=("gradient-descent", "cnn", "resnet", "gradient-flow", "self-supervised-learning")),
        requested_title="Knowledge Distillation", assigned_topic_id="knowledge-distillation", catalog=CATALOG,
        source_context={"kind":"youtube", "video_url":"https://youtu.be/abc123?t=609", "video_id":"abc123", "title":"KD", "channel":"x", "attribution":"local", "concept":"Knowledge Distillation", "source_evidence_summary":"Teacher/student outputs.", "timestamp_seconds":[437, 447]})
    assert kd["category"] == "deep_learning" and kd["concept_type"] == "training_mechanism"
    assert kd["prerequisite_topic_ids"] == kd["related_topic_ids"] == []
    assert kd["sources"][0]["url"] == "https://www.youtube.com/watch?v=abc123"
    assert kd["source_provenance"]["source_derived"]["timestamp_seconds"] == [437, 447]
    assert blocking and "suspicious quote" in blocking[0]["message"]

    clip, _, _ = normalize_topic(topic("CLIP", relations=("covariance", "cnn", "resnet", "gradient-flow")), requested_title="CLIP", assigned_topic_id="clip", catalog=CATALOG)
    assert clip["category"] == "representation_learning" and clip["prerequisite_topic_ids"] == [] and clip["related_topic_ids"] == []

    contrastive, _, _ = normalize_topic(topic("Contrastive Learning", relations=("cnn", "pca", "resnet", "gradient-descent"), latex="z_i^T z_j"), requested_title="Contrastive Learning", assigned_topic_id="contrastive-learning", catalog=CATALOG)
    assert contrastive["category"] == "representation_learning" and contrastive["related_topic_ids"] == []


class QualityProvider:
    def __init__(self, remaining=False): self.calls=[]; self.remaining=remaining

    def structured(self, **kwargs):
        self.calls.append(kwargs["schema_name"])
        if kwargs["schema_name"] == "ultimate_ml_topic_draft":
            generated=topic("CLIP", relations=("covariance", "cnn", "resnet", "gradient-flow"))
            generated["core_explanation"]="Negatives are universally required to avoid collapse."
            return ProviderResult(generated, 100, 120)
        assert kwargs["schema_name"] == "ultimate_ml_topic_quality_review"
        corrected = topic("CLIP", category="representation_learning", relations=(), latex=r"\ell_{i,j}=s\, z_i^\top t_j")
        corrected.update({"concept_type":"named_method", "tags":["multimodal", "contrastive learning"],
            "core_explanation":"CLIP normalizes image and text embeddings, uses symmetric image-to-text and text-to-image batch losses, and learns a positive logit scale (inverse temperature)."})
        return ProviderResult({"corrected_topic": corrected, "quality_report": {"confidence":"high", "blocking_issues_fixed":[{"area":"mathematics", "message":"Restored CLIP-specific normalized embeddings and learned logit scale."}], "blocking_issues_remaining":[{"area":"mathematics", "message":"A human must verify a source-specific numerical claim."}] if self.remaining else []}}, 80, 90)


def test_two_pass_pipeline_repairs_clip_records_separate_costs_and_never_uses_model_id(tmp_path, monkeypatch):
    provider=QualityProvider()
    db=main.Database(tmp_path / "quality.db")
    monkeypatch.setattr(main, "database", db)
    monkeypatch.setattr(main, "ai_service", AIService(db, provider, api_key="fake"))
    response=TestClient(main.app).post("/api/ai/topic-draft", json={"title":"CLIP", "category":"ml_fundamentals", "allow_duplicate":True})
    assert response.status_code == 200
    payload=response.json()["payload"]
    assert payload["quality_status"] == "ready" and payload["category"] == "representation_learning"
    assert payload["related_topic_ids"] == [] and "logit scale" in payload["core_explanation"]
    assert "universally required" not in payload["core_explanation"]
    assert payload["durable_topic_id"] == "clip" and "id" not in payload
    assert provider.calls == ["ultimate_ml_topic_draft", "ultimate_ml_topic_quality_review"]
    operations={group["operation_type"] for group in db.monthly_usage(datetime.now(timezone.utc).strftime("%Y-%m"))[1]}
    assert operations == {"topic_draft", "topic_quality_review"}
    estimate=TestClient(main.app).get("/api/ai/topic-draft-estimate").json()
    assert [item["operation_type"] for item in estimate["operations"]] == ["topic_draft", "topic_quality_review"]


def test_remaining_reviewer_blocker_persists_needs_attention_without_auto_approval(tmp_path, monkeypatch):
    provider=QualityProvider(remaining=True)
    db=main.Database(tmp_path / "quality.db")
    monkeypatch.setattr(main, "database", db)
    monkeypatch.setattr(main, "ai_service", AIService(db, provider, api_key="fake"))
    draft=TestClient(main.app).post("/api/ai/topic-draft", json={"title":"CLIP", "allow_duplicate":True}).json()
    assert draft["state"] == "draft" and draft["payload"]["quality_status"] == "needs_attention"
    assert db.get_draft(draft["id"])["state"] == "draft"


def test_mocked_kd_repair_fixture_requires_stop_gradient_and_temperature_assumptions(tmp_path, monkeypatch):
    class KDProvider:
        def structured(self, **kwargs):
            if kwargs["schema_name"] == "ultimate_ml_topic_draft":
                bad=topic("Knowledge Distillation", relations=("gradient-descent", "self-supervised-learning"), latex=r"\theta_t \leftarrow \operatorname{stopgrad}(\theta_t)")
                bad["core_explanation"]="Temperature softmax has gradient p_s-p_t."
                return ProviderResult(bad, 20, 20)
            fixed=topic("Knowledge Distillation", category="deep_learning", relations=(), latex=r"\frac{\partial L}{\partial z_s}=\frac{1}{T}(p_s^{(T)}-p_t^{(T)})")
            fixed.update({"concept_type":"training_mechanism", "core_explanation":"The teacher branch is stop-gradient: its outputs are treated as constants during the student backward pass. With temperature T in the softmax and the stated loss convention, the gradient includes the corresponding temperature scaling."})
            return ProviderResult({"corrected_topic":fixed,"quality_report":{"confidence":"high","blocking_issues_fixed":[{"area":"mathematics","message":"Corrected stop-gradient semantics and documented temperature scaling."}]}},20,20)
    db=main.Database(tmp_path / "quality.db")
    monkeypatch.setattr(main, "database", db)
    monkeypatch.setattr(main, "ai_service", AIService(db, KDProvider(), api_key="fake"))
    result=TestClient(main.app).post("/api/ai/topic-draft", json={"title":"Knowledge Distillation", "allow_duplicate":True}).json()["payload"]
    assert result["category"] == "deep_learning" and result["prerequisite_topic_ids"] == []
    assert "outputs are treated as constants" in result["core_explanation"] and r"\frac{1}{T}" in result["mathematical_foundation"]["sections"][0]["equations"][0]["latex"]


def test_backend_approval_id_ignores_an_untrusted_payload_id():
    approved=main._approved_topic({"id":"model-invented-id", "title":"Backend Owned Identity", "category":"deep_learning", "difficulty":"intermediate", "quick_recall":"x", "core_explanation":"x"})
    assert approved["id"] == "backend-owned-identity"
