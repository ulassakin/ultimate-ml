"""Scalable metadata resolution regressions; all model interactions are faked."""
from types import SimpleNamespace

import backend.main as main
from backend.ai.metadata_resolution import MetadataResolutionService
from backend.ai.provider import ProviderResult
from backend.ai.service import AIService
from backend.ai.topic_quality import apply_relationship_resolution, metadata_payload_consistency, normalize_topic


def topic(topic_id, title, *, category="deep_learning", concept_type="architecture", summary="Relevant ML concept"):
    return {"id": topic_id, "title": title, "category": category, "concept_type": concept_type,
            "tags": ["representation", "learning"], "one_sentence_summary": summary,
            "difficulty": "intermediate", "quick_recall": "Recall.", "core_explanation": "A bounded explanation."}


class ResolverFake:
    def __init__(self): self.calls=[]
    def structured(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["schema_name"] == "ultimate_ml_relationship_resolution"
        request = __import__("json").loads(kwargs["input_text"])
        assert "topic_catalog" not in request and "retrieved_candidates" in request
        return ProviderResult({"prerequisites": [], "related": [], "rejected_candidates": [
            {"topic_id": request["retrieved_candidates"][0]["topic_id"], "reason": "Similarity alone is not a curriculum edge."}]}, 12, 8)


def test_local_retrieval_caps_candidates_and_cached_resolution_never_repeats_paid_call(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTIMATE_ML_METADATA_TOP_K", "10")
    database = main.Database(tmp_path / "metadata.db")
    catalog = {f"topic-{index}": topic(f"topic-{index}", f"Representation Method {index}") for index in range(30)}
    current = topic("new-method", "New Representation Method", concept_type="named_method")
    service = MetadataResolutionService(database, top_k=10)
    candidates = service.retrieve(current, catalog)
    assert len(candidates) == 10
    assert set(candidates[0]) == {"topic_id", "title", "one_sentence_summary", "concept_type", "category", "tags", "content_hash", "retrieval_score"}
    fake = ResolverFake()
    monkeypatch.setattr(main, "database", database)
    monkeypatch.setattr(main, "library", SimpleNamespace(topics=catalog))
    monkeypatch.setattr(main, "ai_service", AIService(database, fake, api_key="fake"))
    first, _, cost, cached, _, blocking = main._resolve_topic_metadata(current, assigned_topic_id="new-method")
    second, _, second_cost, second_cached, _, second_blocking = main._resolve_topic_metadata(current, assigned_topic_id="new-method")
    assert first["prerequisite_topic_ids"] == first["related_topic_ids"] == []
    assert not blocking and not second_blocking and cost > 0 and not cached
    assert second["metadata_resolution"]["cached"] is True and second_cost == 0 and second_cached
    assert second["prerequisite_topic_ids"] == first["prerequisite_topic_ids"]
    assert len(fake.calls) == 1
    request = __import__("json").loads(fake.calls[0]["input_text"])
    assert len(request["retrieved_candidates"]) == 10


def test_universal_guardrails_reject_documented_generic_edges_without_topic_mapping():
    candidates = [{"topic_id": key, "title": key.replace("-", " ").title()} for key in
                  ("gradient-descent", "cnn", "resnet", "gradient-flow", "knowledge-distillation", "pca", "covariance", "self-attention", "clip")]
    base = topic("vit", "Vision Transformer", category="transformers")
    result, warnings, _ = apply_relationship_resolution(base, {"prerequisites": [
        {"topic_id": "gradient-descent", "reason": "The model is trained with gradient descent.", "confidence": "high"},
        {"topic_id": "cnn", "reason": "CNN locality is a useful comparison.", "confidence": "high"},
        {"topic_id": "self-attention", "reason": "Attention is the defining token-mixing mechanism needed first.", "confidence": "high"}], "related": [
        {"topic_id": "gradient-flow", "reason": "Residual paths affect gradient propagation.", "confidence": "high"},
        {"topic_id": "knowledge-distillation", "reason": "Teacher-student pipelines can use this backbone.", "confidence": "high"},
        {"topic_id": "resnet", "reason": "A canonical alternative vision architecture for comparison.", "confidence": "high"},
        {"topic_id": "clip", "reason": "A central image-encoder pairing and multimodal extension.", "confidence": "high"}], "rejected_candidates": []}, candidates=candidates, assigned_topic_id="vit")
    assert result["prerequisite_topic_ids"] == ["self-attention"]
    assert result["related_topic_ids"] == ["clip", "resnet"]
    assert len(warnings) == 4


def test_kd_clip_and_contrastive_generic_failure_patterns_all_resolve_to_sparse_empty_metadata():
    cases = [
        ("Knowledge Distillation", "training_mechanism", [("gradient-descent", "The student is trained with gradient descent."), ("self-supervised-learning", "It is useful general background for representation learning."), ("cnn", "Teacher and student can use CNN backbones.")]),
        ("CLIP", "named_method", [("covariance", "Its embedding vectors have covariance."), ("resnet", "A ResNet can be its image backbone."), ("cnn", "It uses neural networks for images.")]),
        ("Contrastive Learning", "loss_or_objective", [("gradient-descent", "The objective is trained by gradient descent."), ("pca", "It learns embedding features like PCA."), ("resnet", "ResNet is a possible encoder backbone.")]),
    ]
    for title, concept_type, edges in cases:
        candidates = [{"topic_id": topic_id, "title": topic_id} for topic_id, _ in edges]
        resolved, _, _ = apply_relationship_resolution(topic("new", title, concept_type=concept_type), {"prerequisites": [
            {"topic_id": topic_id, "reason": reason, "confidence": "high"} for topic_id, reason in edges], "related": [], "rejected_candidates": []}, candidates=candidates, assigned_topic_id="new")
        assert resolved["prerequisite_topic_ids"] == []


def test_report_payload_mismatch_is_a_blocker_before_ready():
    payload = topic("new", "Fresh Topic")
    resolved, _, _ = apply_relationship_resolution(payload, {"prerequisites": [], "related": [], "rejected_candidates": []}, candidates=[], assigned_topic_id="new")
    _, topic_hash, _ = MetadataResolutionService.cache_key(resolved, [])
    resolved["metadata_resolution"]["topic_hash"] = topic_hash
    assert metadata_payload_consistency(resolved, {}, catalog=[], assigned_topic_id="new") == []
    resolved["category"] = "representation_learning"
    issues = metadata_payload_consistency(resolved, {"blocking_issues_fixed": [{"area": "taxonomy", "message": "Category repaired."}]}, catalog=[], assigned_topic_id="new")
    assert any(issue["area"] == "taxonomy" for issue in issues)


def test_math_prerequisites_stay_controlled_and_specialized_section_math_is_not_global():
    payload = topic("ssl", "Self-Supervised Learning", concept_type="broad_concept")
    payload["mathematical_foundation"] = {"overview": "Several objectives use different mathematics.", "prerequisites": ["Covariance", "CNN", "Gradient Descent"], "sections": [
        {"title": "Invariance", "explanation": "A contrastive softmax objective.", "equations": [{"latex": r"\ell=-\log \operatorname{softmax}(s/\tau)", "explanation": "Contrastive loss."}]},
        {"title": "Redundancy", "explanation": "Covariance can measure redundant features.", "equations": [{"latex": r"C=\operatorname{Cov}(z)", "explanation": "Covariance matrix."}]},
    ]}
    normalized, warnings, blocking = normalize_topic(payload, requested_title="Self-Supervised Learning", assigned_topic_id="ssl", catalog=[])
    assert not blocking and "Covariance" not in normalized["mathematical_foundation"]["prerequisites"]
    assert "Covariance" in normalized["mathematical_foundation"]["sections"][1]["prerequisites"]
    assert all(value not in {"CNN", "Gradient Descent"} for value in normalized["mathematical_foundation"]["prerequisites"])
    assert warnings
