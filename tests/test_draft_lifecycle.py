"""Safe delete/restart lifecycle controls use isolated data and fake generation."""
from fastapi import HTTPException

import backend.main as main
from tests.test_draft_action_reliability import isolated_client


def broken_payload(title="Vectors and Vector Spaces"):
    return {"title": title, "category": "Mathematical Foundations", "difficulty": "beginner",
        "tags": ["vectors", "linear algebra"], "one_sentence_summary": "A vector-space draft.",
        "quick_recall": "Vectors represent direction and magnitude.", "core_explanation": "Vector spaces organize linear combinations.",
        "quality_status": "needs_attention", "quality_report": {"blocking_issues_remaining": [{"area": "taxonomy", "message": "Legacy category needs correction."}]},
        "generation_metadata": {"user_focus": "Build geometric intuition for ML embeddings."}}


def create_incomplete(database, draft_id="broken", metadata=None):
    database.create_draft(draft_id, "topic", "Vectors and Vector Spaces", broken_payload(), metadata or {})


def test_delete_removes_only_selected_active_draft_and_preserves_history(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    create_incomplete(database, "delete-me")
    database.create_draft("keep-question", "question", "Question", {"questions": []})
    database.review("existing-question", "good")
    history = database.progress()

    deleted = client.delete("/api/ai/drafts/delete-me")
    assert deleted.status_code == 200
    assert database.get_draft("delete-me") is None
    assert database.get_draft("keep-question") is not None
    assert database.progress() == history
    assert client.delete("/api/ai/drafts/delete-me").status_code == 404

    # Video queue/import projections survive, but no longer point at a deleted draft.
    create_incomplete(database, "video-draft", {"youtube_import_id": "video-import", "youtube_concept": "Vectors"})
    database.create_queue_item("queue", youtube_import_id="video-import", canonical_concept="vectors",
        concept={"canonical_name": "Vectors"}, action="create", status="ready", draft_id="video-draft")
    assert client.delete("/api/ai/drafts/video-draft").status_code == 200
    queue = database.get_queue_item("queue")
    assert queue["draft_id"] is None and queue["status"] == "discarded"

    database.create_draft("approved", "topic", "Approved", broken_payload("Approved"))
    database.update_draft("approved", broken_payload("Approved"), "approved")
    assert client.delete("/api/ai/drafts/approved").status_code == 409


def test_restart_recovers_saved_original_inputs_once_and_archives_old_draft(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    input_metadata = {"generation_input": {"title": "Vectors and Vector Spaces", "category": "mathematical_foundations",
        "difficulty": "beginner", "depth": "deep", "tags": ["vectors", "linear algebra"], "focus": "Original focus",
        "include_mathematics": True, "include_examples": False, "include_misconceptions": True,
        "suggest_related_topics": True, "allow_duplicate": False}}
    create_incomplete(database, metadata=input_metadata)
    calls = []

    def fresh(body, **kwargs):
        calls.append((body, kwargs))
        payload = broken_payload(body.title)
        payload.update({"category": body.category, "difficulty": body.difficulty.value, "tags": body.tags,
                        "quality_status": "ready", "quality_report": {"blocking_issues_remaining": []}})
        database.create_draft("fresh", "topic", body.title, payload, {"generation_input": input_metadata["generation_input"]})
        return {"id": "fresh", "state": "draft", "payload": payload, "usage": {"estimated_cost_usd": 0}}

    monkeypatch.setattr(main, "_generate_topic_draft", fresh)
    estimate = client.get("/api/ai/drafts/broken/restart-estimate")
    assert estimate.status_code == 200 and estimate.json()["generation_input"]["depth"] == "deep"
    restarted = client.post("/api/ai/drafts/broken/restart")
    assert restarted.status_code == 200 and restarted.json()["draft"]["id"] == "fresh"
    assert len(calls) == 1
    body, kwargs = calls[0]
    assert body.focus == "Original focus" and body.include_examples is False and body.depth.value == "deep"
    assert kwargs["duplicate_check"] is False
    assert database.get_draft("broken")["state"] == "discarded"
    assert database.get_draft("fresh")["state"] == "draft"
    assert client.post("/api/ai/drafts/broken/restart").status_code == 409


def test_restart_failure_or_budget_block_keeps_original_draft(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    create_incomplete(database)
    original_generate = main._generate_topic_draft
    monkeypatch.setattr(main, "_generate_topic_draft", lambda *args, **kwargs: (_ for _ in ()).throw(HTTPException(502, "provider failed")))
    assert client.post("/api/ai/drafts/broken/restart").status_code == 502
    assert database.get_draft("broken")["state"] == "draft"

    monkeypatch.setattr(main, "_generate_topic_draft", original_generate)
    main.ai_service.update_settings({"monthly_budget_usd": 0})
    blocked = client.post("/api/ai/drafts/broken/restart")
    assert blocked.status_code == 429
    assert database.get_draft("broken")["state"] == "draft"


def test_restart_reports_missing_video_context_without_removing_draft(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    create_incomplete(database, metadata={"youtube_import_id": "missing", "youtube_concept": "Vectors"})
    response = client.post("/api/ai/drafts/broken/restart")
    assert response.status_code == 422
    assert "video import" in response.json()["detail"]
    assert database.get_draft("broken")["state"] == "draft"
    assert client.post("/api/ai/drafts/missing/restart").status_code == 404


def test_restart_and_delete_frontend_contract_has_confirmation_and_single_flight():
    source = (main.ROOT / "frontend" / "existing-draft-quality-review.js").read_text(encoding="utf-8")
    assert "Delete this draft? This removes the draft only. Approved topics and questions are not affected." in source
    assert "restart-estimate" in source and "Restart generation" in source
    assert "draftLifecycleBusy" in source and "finally" in source
    assert "method:'DELETE'" in source and "await draftQueue()" in source
