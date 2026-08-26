"""DINO-style persisted state regression; no provider call is permitted."""
from copy import deepcopy

from backend.ai.metadata_resolution import MetadataResolutionService

from tests.test_draft_action_reliability import isolated_client


def dino_payload():
    payload = {
        "title": "DINO Editor State Regression", "category": "deep_learning", "difficulty": "advanced",
        "concept_type": "named_method", "tags": ["self-supervised learning"],
        "one_sentence_summary": "A student-teacher self-supervised visual representation method.",
        "quick_recall": "A student matches a slowly updated teacher across views.",
        "core_explanation": "The teacher produces targets and the student learns to match them without labels.",
        "mathematical_foundation": {"overview": "The matching objective requires gradients.", "prerequisites": ["Gradients"], "sections": []},
        "prerequisite_topic_ids": [], "related_topic_ids": [], "relationship_justifications": [],
        "quality_status": "ready", "quality_report": {"confidence": "high", "blocking_issues_remaining": []},
    }
    _, topic_hash, _ = MetadataResolutionService.cache_key(payload, [])
    payload["metadata_resolution"] = {
        "resolver_version": "relationship-resolver-v1", "resolved_category": "deep_learning",
        "resolved_concept_type": "named_method", "topic_hash": topic_hash,
        "durable_edges": {"prerequisites": [], "related": []},
        "math_prerequisites": {"global": ["Gradients"], "sections": {}},
    }
    return payload


def test_dino_advanced_json_restore_save_reload_validate_and_approve(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    original = dino_payload()
    database.create_draft("dino-editor", "topic", original["title"], original)

    # The user first removes the resolver-owned math prerequisite: validation
    # correctly reports a mismatch while preserving the draft.
    removed = deepcopy(original)
    removed["mathematical_foundation"]["prerequisites"] = []
    saved_removed = client.put("/api/ai/drafts/dino-editor", json={"payload": removed})
    assert saved_removed.status_code == 200
    invalid = client.get("/api/ai/drafts/dino-editor/validate").json()
    assert invalid["valid"] is False
    assert any("mathematical prerequisites disagree" in error["message"] for error in invalid["errors"])

    # This is the exact object the Advanced JSON editor applies and then saves.
    restored = deepcopy(saved_removed.json()["payload"])
    restored["mathematical_foundation"]["prerequisites"] = ["Gradients"]
    saved_restored = client.put("/api/ai/drafts/dino-editor", json={"payload": restored})
    assert saved_restored.status_code == 200
    persisted = saved_restored.json()
    assert persisted["payload"]["mathematical_foundation"]["prerequisites"] == ["Gradients"]
    assert client.get("/api/ai/drafts/dino-editor").json()["payload"] == persisted["payload"]
    assert client.get("/api/ai/drafts/dino-editor/validate").json()["valid"] is True

    approved = client.post("/api/ai/drafts/dino-editor/approve")
    assert approved.status_code == 200
    assert approved.json()["topic"]["id"] == "dino-editor-state-regression"
    assert database.monthly_usage("2026-08")[0] == 0
