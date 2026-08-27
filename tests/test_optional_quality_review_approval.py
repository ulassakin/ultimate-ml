"""Approval stays local when optional quality review has not run or failed."""

from copy import deepcopy

import backend.main as main

from tests.test_draft_action_reliability import isolated_client, ready_payload


def create_topic_draft(database, draft_id, payload):
    database.create_draft(draft_id, "topic", payload["title"], payload)


def unreviewed_payload(title="Weights and Biases"):
    payload = ready_payload() | {"title": title, "category": "ml_fundamentals", "concept_type": "broad_concept"}
    payload.pop("quality_report", None)
    payload.update({"quality_review_state": "not_run", "quality_status": "not_reviewed"})
    return payload


def test_valid_not_run_draft_approves_without_provider_call(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = unreviewed_payload()
    create_topic_draft(database, "weights", payload)
    monkeypatch.setattr(main.ai_service, "generate", lambda **_: (_ for _ in ()).throw(AssertionError("Approval must not call a provider")))

    validation = client.get("/api/ai/drafts/weights/validate").json()
    assert validation["valid"] is True
    assert validation["quality_review_recommendation"]["level"] == "low"
    approved = client.post("/api/ai/drafts/weights/approve")
    assert approved.status_code == 200
    assert approved.json()["topic"]["id"] == "weights-and-biases"
    assert database.monthly_usage(main.datetime.now(main.timezone.utc).strftime("%Y-%m"))[0] == 0


def test_reviewed_ready_draft_remains_approvable(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = ready_payload() | {"title": "Reviewed Ready Topic"}
    create_topic_draft(database, "reviewed-ready", payload)

    assert client.get("/api/ai/drafts/reviewed-ready/validate").json()["valid"] is True
    assert client.post("/api/ai/drafts/reviewed-ready/approve").status_code == 200


def test_completed_review_with_real_blockers_still_blocks_approval(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = unreviewed_payload("Blocked Topic") | {
        "quality_review_state": "reviewed", "quality_status": "needs_attention",
        "quality_report": {"blocking_issues_remaining": [{"area": "mathematics", "message": "Equation is malformed."}]},
    }
    create_topic_draft(database, "blocked", payload)

    validation = client.get("/api/ai/drafts/blocked/validate").json()
    assert validation["valid"] is False
    assert validation["errors"] == [{"field": "quality_review", "message": "Resolve the quality-review blocking issues before approval."}]
    assert client.post("/api/ai/drafts/blocked/approve").status_code == 422


def test_failed_review_is_warning_not_an_approval_gate(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = unreviewed_payload("Failed Review Topic") | {"quality_review_state": "failed", "quality_status": "not_reviewed"}
    create_topic_draft(database, "failed-review", payload)

    validation = client.get("/api/ai/drafts/failed-review/validate").json()
    assert validation["valid"] is True
    assert "failed" in validation["warnings"][0]["message"].casefold()
    assert client.post("/api/ai/drafts/failed-review/approve").status_code == 200


def test_local_risk_heuristic_recommends_review_for_pca_but_not_basic_concept(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    low = unreviewed_payload("Basic Concepts")
    high = unreviewed_payload("Principal Component Analysis") | {
        "category": "classical_ml", "concept_type": "mathematical_concept",
        "mathematical_foundation": {"overview": "PCA uses a covariance eigendecomposition.", "prerequisites": [], "sections": [
            {"title": "Projection", "explanation": "A long-enough explanation of projection and variance.", "equations": [
                {"latex": "z = W^T x", "explanation": "Project x onto components."},
                {"latex": "C = X^T X / n", "explanation": "The covariance-like matrix."},
            ]}
        ]},
    }
    create_topic_draft(database, "low", low)
    create_topic_draft(database, "pca", high)

    assert client.get("/api/ai/drafts/low").json()["quality_review_recommendation"]["recommended"] is False
    recommendation = client.get("/api/ai/drafts/pca").json()["quality_review_recommendation"]
    assert recommendation["recommended"] is True
    assert recommendation["level"] == "high"
    assert "mathematical concept" in recommendation["reasons"]


def test_existing_approved_topics_are_unchanged_by_optional_approval(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    before = deepcopy(main.library.topics["dino"])
    create_topic_draft(database, "new-topic", unreviewed_payload("New Optional Approval Topic"))

    assert client.post("/api/ai/drafts/new-topic/approve").status_code == 200
    assert main.library.topics["dino"] == before


def test_draft_review_ui_exposes_optional_review_confirmation_once():
    source = (main.ROOT / "frontend" / "simplified-topic-workflow.js").read_text(encoding="utf-8")
    assert "Review risk: <strong>Low</strong>" in source
    assert "Quality review is recommended before approval" in source
    assert "This topic has not been quality-reviewed yet. You can approve it now or run Quality Review first." in source
    assert "Approve anyway" in source and "Run Quality Review first" in source
    assert "if(!confirmation.hidden)return;" in source
