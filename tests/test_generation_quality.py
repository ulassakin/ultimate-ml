"""Regression fixtures for mistakes observed in paid topic generations; all calls are faked."""
import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.provider import ProviderResult
from backend.ai.schemas import TopicDraft
from backend.ai.service import AIService
from backend.ai.topic_quality import apply_relationship_resolution, normalize_topic, relationship_lint


CATALOG = [
    {"id": item, "title": item.replace("-", " ").title(), "category": "deep_learning", "summary": "Existing topic."}
    for item in ("gradient-descent", "cnn", "resnet", "gradient-flow", "self-supervised-learning", "covariance", "pca", "self-attention", "transformer-encoder", "contrastive-language-image-pretraining-clip", "knowledge-distillation")
]


def topic(title, *, category="ml_fundamentals", relations=(), latex="z_s = x\""):
    return {"title": title, "category": category, "difficulty": "intermediate", "concept_type": "broad_concept",
        "tags": ["self-supervised learning", "representation learning"], "one_sentence_summary": "A concise summary.",
        "quick_recall": "A concise recall.", "core_explanation": "A technically bounded explanation.",
        "prerequisite_topic_ids": list(relations), "related_topic_ids": list(relations),
        "relationship_justifications": [{"topic_id": item, "relationship": "prerequisite", "reason": "This exists, so it is supposedly useful.", "confidence": "high"} for item in relations],
        "mathematical_foundation": {"overview": "Math.", "sections": [{"title": "Objective", "explanation": "Defined symbols.", "equations": [{"latex": latex, "explanation": "A formula explanation."}]}]},
        "suggested_new_topic_relationships": [{"title": "Missing Neighbor", "relationship": "related", "reason": "A genuinely useful absent concept."}]}


def test_normalization_is_title_agnostic_and_canonicalizes_provenance_and_bad_math():
    kd, _, blocking = normalize_topic(topic("Knowledge Distillation", relations=("gradient-descent", "cnn", "resnet", "gradient-flow", "self-supervised-learning")),
        requested_title="Knowledge Distillation", assigned_topic_id="knowledge-distillation", catalog=CATALOG,
        source_context={"kind":"youtube", "video_url":"https://youtu.be/abc123?t=609", "video_id":"abc123", "title":"KD", "channel":"x", "attribution":"local", "concept":"Knowledge Distillation", "source_evidence_summary":"Teacher/student outputs.", "timestamp_seconds":[437, 447]})
    assert kd["category"] == "ml_fundamentals" and kd["concept_type"] == "broad_concept"
    assert kd["sources"][0]["url"] == "https://www.youtube.com/watch?v=abc123"
    assert kd["source_provenance"]["source_derived"]["timestamp_seconds"] == [437, 447]
    assert blocking and "suspicious quote" in blocking[0]["message"]

    resolved, warnings, _ = apply_relationship_resolution(kd, {"prerequisites":[{"topic_id":"gradient-descent","reason":"It trains the model with an optimizer.","confidence":"high"}], "related":[{"topic_id":"cnn","reason":"It can use a neural-network backbone.","confidence":"high"}], "rejected_candidates":[]}, candidates=[{"topic_id":"gradient-descent","title":"Gradient Descent"},{"topic_id":"cnn","title":"CNN"}], assigned_topic_id="knowledge-distillation")
    assert resolved["prerequisite_topic_ids"] == resolved["related_topic_ids"] == []
    assert warnings


def test_vit_relationship_fixture_keeps_only_strong_high_confidence_curriculum_edges():
    vit=topic("Vision Transformer (ViT)", category="transformers", relations=(), latex="z_i^Tz_j")
    vit.update({"concept_type":"architecture", "prerequisite_topic_ids":["cnn","gradient-descent","self-attention","transformer-encoder"],
        "related_topic_ids":["knowledge-distillation","gradient-flow","resnet","contrastive-language-image-pretraining-clip"],
        "relationship_justifications":[
            {"topic_id":"cnn","relationship":"prerequisite","confidence":"high","reason":"CNN locality is a useful architectural comparison."},
            {"topic_id":"gradient-descent","relationship":"prerequisite","confidence":"high","reason":"ViT is trained with gradient-based optimization."},
            {"topic_id":"self-attention","relationship":"prerequisite","confidence":"high","reason":"Self-attention is the core token-mixing mechanism that a learner needs first."},
            {"topic_id":"transformer-encoder","relationship":"prerequisite","confidence":"high","reason":"ViT is a transformer encoder applied to image patch tokens."},
            {"topic_id":"knowledge-distillation","relationship":"related","confidence":"high","reason":"Teacher-student vision pipelines sometimes use ViT backbones."},
            {"topic_id":"gradient-flow","relationship":"related","confidence":"high","reason":"Residual pathways influence gradient propagation during training."},
            {"topic_id":"resnet","relationship":"related","confidence":"high","reason":"ResNet is the canonical convolutional architectural comparison for ViT."},
            {"topic_id":"contrastive-language-image-pretraining-clip","relationship":"related","confidence":"high","reason":"CLIP commonly uses ViT as a major image encoder in a central multimodal pairing."},
        ]})
    normalized, _, blocking=normalize_topic(vit, requested_title="Vision Transformer (ViT)", assigned_topic_id="vision-transformer-vit", catalog=CATALOG)
    normalized, warnings, _=apply_relationship_resolution(normalized, {"prerequisites":[
        {"topic_id":"cnn","reason":"CNN locality is a useful architectural comparison.","confidence":"high"},
        {"topic_id":"gradient-descent","reason":"ViT is trained with gradient-based optimization.","confidence":"high"},
        {"topic_id":"self-attention","reason":"Self-attention is the core token-mixing mechanism that a learner needs first.","confidence":"high"},
        {"topic_id":"transformer-encoder","reason":"ViT is a transformer encoder applied to image patch tokens.","confidence":"high"}],
        "related":[
        {"topic_id":"knowledge-distillation","reason":"Teacher-student vision pipelines sometimes use ViT backbones.","confidence":"high"},
        {"topic_id":"gradient-flow","reason":"Residual pathways influence gradient propagation during training.","confidence":"high"},
        {"topic_id":"resnet","reason":"ResNet is the canonical convolutional architectural comparison for ViT.","confidence":"high"},
        {"topic_id":"contrastive-language-image-pretraining-clip","reason":"CLIP commonly uses ViT as a major image encoder in a central multimodal pairing.","confidence":"high"}], "rejected_candidates":[]},
        candidates=[{"topic_id":item["id"],"title":item["title"]} for item in CATALOG], assigned_topic_id="vision-transformer-vit")
    assert normalized["prerequisite_topic_ids"] == ["self-attention", "transformer-encoder"]
    assert normalized["related_topic_ids"] == ["contrastive-language-image-pretraining-clip", "resnet"]
    assert not blocking and len(normalized["relationship_justifications"]) == 4
    assert any("generic" in warning["message"] for warning in warnings)


def test_relationship_confidence_and_density_lint_moves_medium_to_suggestions_and_blocks_weak_edges():
    payload=topic("A New Architecture", relations=(), latex="x+y")
    payload.update({"concept_type":"architecture", "prerequisite_topic_ids":["self-attention"], "related_topic_ids":["resnet", "cnn"],
        "relationship_justifications":[
            {"topic_id":"self-attention","relationship":"prerequisite","confidence":"high","reason":"The architecture directly uses attention as its defining computation."},
            {"topic_id":"resnet","relationship":"related","confidence":"medium","reason":"It can be compared as an older vision architecture."},
            {"topic_id":"cnn","relationship":"related","confidence":"low","reason":"Both are neural networks for images."},
        ]})
    normalized, warnings, _=normalize_topic(payload, requested_title="A New Architecture", assigned_topic_id="a-new-architecture", catalog=CATALOG)
    assert normalized["prerequisite_topic_ids"] == ["self-attention"] and normalized["related_topic_ids"] == []
    assert normalized["suggested_new_topic_relationships"][1]["title"] == "Resnet"
    errors, density_warnings=relationship_lint({**payload,"related_topic_ids":["resnet"]}, catalog=CATALOG, assigned_topic_id="a-new-architecture")
    assert any("high-confidence" in item["message"] for item in errors) and density_warnings == []


class QualityProvider:
    def __init__(self, remaining=False): self.calls=[]; self.remaining=remaining

    def structured(self, **kwargs):
        self.calls.append(kwargs["schema_name"])
        if kwargs["schema_name"] == "ultimate_ml_topic_draft":
            generated=topic("CLIP", relations=("covariance", "cnn", "resnet", "gradient-flow"))
            generated["core_explanation"]="Negatives are universally required to avoid collapse."
            return ProviderResult(generated, 100, 120)
        if kwargs["schema_name"] == "ultimate_ml_relationship_resolution":
            request=json.loads(kwargs["input_text"])
            assert "topic_catalog" not in request
            return ProviderResult({"prerequisites":[],"related":[],"rejected_candidates":[]}, 15, 10)
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
    assert provider.calls == ["ultimate_ml_topic_draft", "ultimate_ml_topic_quality_review", "ultimate_ml_relationship_resolution"]
    operations={group["operation_type"] for group in db.monthly_usage(datetime.now(timezone.utc).strftime("%Y-%m"))[1]}
    assert operations == {"topic_draft", "topic_quality_review", "metadata_relationship_resolution"}
    estimate=TestClient(main.app).get("/api/ai/topic-draft-estimate").json()
    assert [item["operation_type"] for item in estimate["operations"]] == ["topic_draft", "topic_quality_review", "metadata_relationship_resolution"]


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
            if kwargs["schema_name"] == "ultimate_ml_relationship_resolution":
                return ProviderResult({"prerequisites":[],"related":[],"rejected_candidates":[]}, 5, 5)
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
