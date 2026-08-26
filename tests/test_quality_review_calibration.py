"""Calibration regressions are deterministic and make no provider/API calls."""
import json

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.provider import ProviderResult
from backend.ai.schemas import TopicDraft
from backend.ai.service import AIService
from backend.ai.topic_quality import final_quality_report, normalize_topic


CATALOG=[]


def simclr_payload():
    return {"title":"SimCLR", "category":"representation_learning", "difficulty":"intermediate", "concept_type":"named_method",
        "one_sentence_summary":"Contrastive visual representation learning.", "quick_recall":"Augment, encode, project, and contrast views.",
        "core_explanation":"SimCLR uses augmentations, a projection head, and a contrastive NT-Xent objective to learn representations.",
        "mechanism":"Two augmented views are encoded, projected, and optimized with a contrastive loss.",
        "mathematical_foundation":{"overview":"A contrastive objective.", "prerequisites":["gradient-descent","cnn"], "sections":[
            {"title":"NT-Xent", "explanation":"A softmax contrastive loss.", "equations":[{"latex":r"\ell_{i,j}=-s_{i,j}/\tau", "explanation":"A positive is contrasted with batch candidates."}]}]},
        "sources":[{"title":"Short video", "type":"youtube", "url":"https://www.youtube.com/watch?v=abc123"}],
        "source_provenance":{"source_derived":{"kind":"youtube", "video_url":"https://www.youtube.com/watch?v=abc123", "video_id":"abc123", "title":"Short video", "source_evidence_summary":"The video introduces SimCLR contrastive training.", "timestamp_seconds":[12,24]}, "ai_expanded":"The educational explanation was AI-expanded from the selected concept and requires human review."}}


def test_honest_short_source_is_ready_with_warnings_and_simclr_math_is_precise():
    normalized, warnings, blocking=normalize_topic(simclr_payload(), requested_title="SimCLR", assigned_topic_id="simclr", catalog=CATALOG)
    assert not blocking
    assert "cnn" not in normalized["mathematical_foundation"]["prerequisites"]
    assert "gradient-descent" not in normalized["mathematical_foundation"]["prerequisites"]
    assert "Softmax" in normalized["mathematical_foundation"]["prerequisites"]
    assert "Contrastive learning intuition" in normalized["mathematical_foundation"]["prerequisites"]
    report=final_quality_report({"confidence":"high", "blocking_issues_remaining":[{"area":"source_grounding","message":"The short source segment has limited evidence for the AI-expanded explanation."}]}, warnings, blocking, payload=normalized)
    assert report["blocking_issues_remaining"] == []
    assert any(item["area"] == "source_grounding" for item in report["warnings"])
    assert any(item["area"] == "named_method_completeness" for item in report["warnings"])


def test_false_source_attribution_and_provenance_contradiction_stay_blocking():
    normalized, warnings, blocking=normalize_topic(simclr_payload(), requested_title="SimCLR", assigned_topic_id="simclr", catalog=CATALOG)
    report=final_quality_report({"confidence":"high", "blocking_issues_remaining":[{"area":"source_grounding","message":"An unsupported claim is falsely presented as source-derived."}]}, warnings, blocking, payload=normalized)
    assert report["blocking_issues_remaining"][0]["area"] == "source_grounding"

    contradictory=simclr_payload()
    contradictory["source_provenance"]["source_derived"]["video_id"]="other-video"
    _, _, contradiction_blocks=normalize_topic(contradictory, requested_title="SimCLR", assigned_topic_id="simclr", catalog=CATALOG)
    assert any("contradicts" in item["message"] for item in contradiction_blocks)
    timestamp_conflict=simclr_payload()
    timestamp_conflict["source_provenance"]["source_derived"]["source_evidence_summary"]="At 609s the video introduces SimCLR."
    _, _, timestamp_blocks=normalize_topic(timestamp_conflict, requested_title="SimCLR", assigned_topic_id="simclr", catalog=CATALOG)
    assert any("timestamp contradicts" in item["message"] for item in timestamp_blocks)


def test_simclr_missing_core_mechanism_is_blocking_but_symmetric_wording_is_only_warning():
    incomplete=simclr_payload()
    incomplete["core_explanation"]="SimCLR learns representations."
    incomplete["mechanism"]="It trains a network."
    _, warnings, blocking=normalize_topic(incomplete, requested_title="SimCLR", assigned_topic_id="simclr", catalog=CATALOG)
    assert any(item["area"] == "named_method_completeness" for item in blocking)
    assert any(item["area"] == "named_method_completeness" for item in warnings)


def test_existing_short_source_review_stays_reviewed_with_warnings_and_no_generator_call(tmp_path, monkeypatch):
    class Reviewer:
        def __init__(self): self.calls=[]
        def structured(self, **kwargs):
            self.calls.append(kwargs["schema_name"])
            if kwargs["schema_name"] == "ultimate_ml_relationship_resolution":
                return ProviderResult({"prerequisites":[],"related":[],"rejected_candidates":[]}, 5, 5)
            corrected={key:value for key,value in simclr_payload().items() if key in TopicDraft.model_fields}
            return ProviderResult({"corrected_topic":corrected,"quality_report":{"confidence":"high","blocking_issues_remaining":[{"area":"source_grounding","message":"The short source segment has limited evidence for AI-expanded context."}]}}, 10, 20)
    database=main.Database(tmp_path / "quality.db")
    database.create_draft("simclr", "topic", "SimCLR", simclr_payload(), {"request_cost_usd":0.01})
    reviewer=Reviewer()
    monkeypatch.setattr(main, "database", database)
    monkeypatch.setattr(main, "ai_service", AIService(database, reviewer, api_key="fake"))
    result=TestClient(main.app).post("/api/ai/drafts/simclr/quality-review", json={})
    assert result.status_code == 200
    payload=database.get_draft("simclr")["payload"]
    assert payload["existing_quality_review"]["status"] == "reviewed"
    assert payload["quality_report"]["blocking_issues_remaining"] == []
    assert reviewer.calls == ["ultimate_ml_topic_quality_review", "ultimate_ml_relationship_resolution"]
