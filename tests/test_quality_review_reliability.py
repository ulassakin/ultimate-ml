"""Focused-review consistency and cost regressions; all providers are fakes."""
import json
import math
from copy import deepcopy

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.provider import ProviderResult
from backend.ai.pricing import DEFAULT_MODEL, estimate_cost
from backend.ai.service import AIService
from backend.ai.topic_quality import final_quality_report


def pca_payload(equation=r"\Sigma_{աբ?} = \mathrm{Cov}(X_a, X_b)"):
    return {
        "title": "Principal Component Analysis",
        "category": "classical_ml",
        "difficulty": "intermediate",
        "concept_type": "mathematical_concept",
        "tags": ["linear algebra", "dimensionality reduction"],
        "one_sentence_summary": "PCA finds high-variance orthogonal directions.",
        "quick_recall": "Project onto directions that preserve variance.",
        "core_explanation": "PCA rotates data into orthogonal directions ordered by variance.",
        "mathematical_foundation": {
            "overview": "Covariance describes co-variation.",
            "prerequisites": ["Covariance"],
            "sections": [{"title": "Covariance", "explanation": "Covariance entries compare two coordinates.",
                          "equations": [{"latex": equation, "explanation": "The covariance between coordinates a and b."}]}],
        },
    }


class Reviewer:
    def __init__(self, response):
        self.response, self.calls, self.requests = response, [], []

    def structured(self, **kwargs):
        self.calls.append(kwargs["schema_name"])
        self.requests.append(kwargs["input_text"])
        return ProviderResult(self.response, 321, 123)


def reviewed_client(tmp_path, monkeypatch, provider, payload=None):
    database = main.Database(tmp_path / "quality-review.db")
    database.create_draft("pca-draft", "topic", "Principal Component Analysis", payload or pca_payload(), {"request_cost_usd": 0.02})
    database.create_draft("question-draft", "question", "Unchanged questions", {"topic_id": "gradient-descent", "questions": []})
    database.review("unchanged-review", "good")
    monkeypatch.setattr(main, "database", database)
    monkeypatch.setattr(main, "ai_service", AIService(database, provider, api_key="fake"))
    return TestClient(main.app), database


def review_response(corrected, *, changes=None, report=None):
    return {"corrected_topic": corrected, "changes": changes or [], "quality_report": report or {"confidence": "high"}}


def test_pca_notation_fix_has_a_backend_derived_diff_and_ready_status(tmp_path, monkeypatch):
    corrected = pca_payload(r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)")
    provider = Reviewer(review_response(corrected, changes=[{
        "field_path": "mathematical_foundation.sections[0].equations[0].latex", "change_type": "replace",
        "reason": "Removed malformed notation.", "old_value": r"\Sigma_{աբ?} = \mathrm{Cov}(X_a, X_b)",
        "new_value": r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)",
    }], report={"confidence": "high", "blocking_issues_fixed": [{"area": "mathematics", "message": "Fixed malformed covariance notation."}]}))
    client, database = reviewed_client(tmp_path, monkeypatch, provider)

    result = client.post("/api/ai/drafts/pca-draft/quality-review", json={})
    assert result.status_code == 200
    response_usage = result.json()["usage"]
    assert response_usage["input_tokens"] == 321 and response_usage["output_tokens"] == 123
    assert response_usage["estimated_cost_usd"] == response_usage["quality_review_estimated_cost_usd"]
    assert response_usage["maximum_estimated_cost_usd"] >= response_usage["estimated_cost_usd"]
    payload = result.json()["draft"]["payload"]
    assert payload["quality_status"] == "ready"
    assert payload["mathematical_foundation"]["sections"][0]["equations"][0]["latex"] == r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)"
    assert payload["quality_report"]["changes"] == [{
        "field_path": "mathematical_foundation.sections[0].equations[0].latex", "change_type": "replace",
        "old_value": r"\Sigma_{աբ?} = \mathrm{Cov}(X_a, X_b)", "new_value": r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)",
    }]
    assert provider.calls == ["ultimate_ml_topic_quality_review"]
    # Compact context contains no graph-era metadata or stored quality history.
    assert "related_topic_ids" not in provider.requests[0] and "quality_report" not in provider.requests[0]
    usage = database.monthly_usage(main.datetime.now(main.timezone.utc).strftime("%Y-%m"))[1]
    quality_usage = next(group for group in usage if group["operation_type"] == "topic_quality_review_existing")
    assert quality_usage["input_tokens"] == 321 and quality_usage["output_tokens"] == 123


def test_false_concept_type_fix_cannot_be_ready_when_payload_is_unchanged(tmp_path, monkeypatch):
    unchanged = pca_payload(r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)")
    unchanged["category"] = "mathematical_foundations"
    unchanged["concept_type"] = "broad_concept"
    provider = Reviewer(review_response(unchanged, changes=[{
        "field_path": "concept_type", "change_type": "replace", "reason": "Taxonomy compatibility.",
        "old_value": "broad_concept", "new_value": "mathematical_concept",
    }], report={"confidence": "high", "blocking_issues_fixed": [{"area": "taxonomy", "message": "Changed concept_type from broad_concept to the taxonomy-compatible value."}]}))
    client, _ = reviewed_client(tmp_path, monkeypatch, provider, payload=unchanged)

    result = client.post("/api/ai/drafts/pca-draft/quality-review", json={})
    assert result.status_code == 200
    payload = result.json()["draft"]["payload"]
    assert payload["concept_type"] == "broad_concept"
    assert payload["quality_status"] == "needs_attention"
    assert payload["quality_report"]["blocking_issues_fixed"] == []
    assert any("concept_type" in issue["message"] for issue in payload["quality_report"]["blocking_issues_remaining"])


def test_valid_no_op_canonical_category_claim_is_discarded_and_ready(tmp_path, monkeypatch):
    valid = pca_payload(r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)")
    valid["category"] = "mathematical_foundations"
    valid["concept_type"] = "mathematical_concept"
    provider = Reviewer(review_response(valid, changes=[{
        "field_path": "category", "change_type": "replace", "reason": "Ensure canonical taxonomy ID.",
        "old_value": "mathematical_foundations", "new_value": "mathematical_foundations",
    }], report={"confidence": "high", "blocking_issues_fixed": [{
        "area": "taxonomy", "message": "Changed category to the canonical taxonomy ID."
    }]}))
    client, _ = reviewed_client(tmp_path, monkeypatch, provider, payload=valid)

    result = client.post("/api/ai/drafts/pca-draft/quality-review", json={})
    payload = result.json()["draft"]["payload"]
    assert result.status_code == 200 and payload["quality_status"] == "ready"
    assert payload["quality_report"]["blocking_issues_remaining"] == []
    assert payload["quality_report"]["changes"] == []
    assert payload["quality_report"]["reviewer_change_claims"] == []


def test_existing_reviewed_no_op_claim_is_reconciled_locally_without_provider_call(tmp_path, monkeypatch):
    valid = pca_payload(r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)")
    valid["category"] = "mathematical_foundations"
    valid["concept_type"] = "mathematical_concept"
    valid.update({
        "quality_review_state": "reviewed", "quality_status": "needs_attention",
        "quality_report": {
            "changes": [],
            "reviewer_change_claims": [{"field_path": "category", "change_type": "replace", "reason": "Ensure canonical taxonomy ID."}],
            "blocking_issues_remaining": [
                {"area": "schema", "message": "Reviewer claimed a change at 'category' that the final payload does not reflect."},
                {"area": "schema", "message": "Quality report claims a fix that the final payload does not reflect: Changed category to the canonical taxonomy ID."},
            ],
        },
    })
    provider = Reviewer(review_response(valid))
    client, database = reviewed_client(tmp_path, monkeypatch, provider, payload=valid)

    reopened = client.get("/api/ai/drafts/pca-draft")
    payload = reopened.json()["payload"]
    assert reopened.status_code == 200 and payload["quality_status"] == "ready"
    assert payload["quality_report"]["blocking_issues_remaining"] == []
    assert payload["quality_report"]["reviewer_change_claims"] == []
    assert provider.calls == [] and database.get_draft("pca-draft")["payload"]["quality_status"] == "ready"


def test_no_op_review_is_ready_without_cosmetic_rewrite(tmp_path, monkeypatch):
    good = pca_payload(r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)")
    provider = Reviewer(review_response(deepcopy(good), changes=[], report={"confidence": "high", "warnings": []}))
    client, database = reviewed_client(tmp_path, monkeypatch, provider, payload=good)
    questions_before, history_before = database.get_draft("question-draft"), database.progress()

    result = client.post("/api/ai/drafts/pca-draft/quality-review", json={})
    payload = result.json()["draft"]["payload"]
    assert result.status_code == 200 and payload["quality_status"] == "ready"
    assert payload["quality_report"]["changes"] == []
    assert payload["core_explanation"] == good["core_explanation"]
    assert database.get_draft("question-draft") == questions_before and database.progress() == history_before


def test_report_deduplicates_and_an_unresolved_duplicate_wins_over_fixed():
    report = final_quality_report({
        "confidence": "high",
        "warnings": [{"area": "mathematics", "message": "Use consistent notation."}, {"area": "mathematics", "message": "Use consistent notation."}],
        "blocking_issues_fixed": [{"area": "schema", "message": "A required field is missing."}],
        "blocking_issues_remaining": [{"area": "schema", "message": "A required field is missing."}],
    }, [], [], payload=pca_payload(r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)"))
    assert len(report["warnings"]) == 1
    assert report["blocking_issues_fixed"] == []
    assert len(report["blocking_issues_remaining"]) == 1


def test_stale_taxonomy_blockers_disappear_after_canonicalization(tmp_path, monkeypatch):
    display_label = pca_payload(r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)")
    display_label["category"] = "Mathematical Foundations"
    display_label["concept_type"] = "mathematical_concept"
    provider = Reviewer(review_response(display_label, report={"confidence": "high", "blocking_issues_remaining": [
        {"area": "taxonomy", "message": "Unknown category 'Mathematical Foundations'."},
        {"area": "taxonomy", "message": "Concept type 'mathematical_concept' is not compatible with category 'Mathematical Foundations'."},
    ]}))
    client, _ = reviewed_client(tmp_path, monkeypatch, provider, payload=display_label)

    result = client.post("/api/ai/drafts/pca-draft/quality-review", json={})
    payload = result.json()["draft"]["payload"]
    assert payload["category"] == "mathematical_foundations"
    assert payload["quality_status"] == "ready"
    assert payload["quality_report"]["blocking_issues_remaining"] == []


def test_compact_review_context_has_a_lower_mocked_token_cost_than_legacy_payload(tmp_path, monkeypatch):
    legacy = pca_payload()
    legacy.update({
        "related_topic_ids": ["cnn", "resnet"],
        "prerequisite_topic_ids": ["covariance"],
        "metadata_resolution": {"obsolete_audit": "x" * 500},
        "quality_report": {"warnings": [{"message": "old report " + "x" * 500}]},
        "generation_metadata": {"historical_detail": "x" * 500},
    })
    corrected = pca_payload(r"\Sigma_{ab} = \mathrm{Cov}(X_a, X_b)")
    provider = Reviewer(review_response(corrected, changes=[{
        "field_path": "mathematical_foundation.sections[0].equations[0].latex",
        "change_type": "replace", "reason": "Correct malformed notation.",
    }]))
    client, _ = reviewed_client(tmp_path, monkeypatch, provider, payload=legacy)

    assert client.post("/api/ai/drafts/pca-draft/quality-review", json={}).status_code == 200
    old_request = {"mode": "existing_paid_draft_quality_review", "taxonomy": main.taxonomy_context(), "candidate": legacy,
                   "source_context": "No source context supplied."}
    old_tokens, compact_tokens = math.ceil(len(json.dumps(old_request)) / 4), math.ceil(len(provider.requests[0]) / 4)
    before = estimate_cost(DEFAULT_MODEL, old_tokens, 123)
    after = estimate_cost(DEFAULT_MODEL, compact_tokens, 123)
    assert compact_tokens < old_tokens and after < before
