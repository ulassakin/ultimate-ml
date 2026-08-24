import json
import shutil

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.service import AIService
from backend.content import load_library


class NoProvider:
    def structured(self, **kwargs):
        raise AssertionError("A local edit or approval must not call the AI provider")


def local_client(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    shutil.copytree(main.ROOT / "content", root / "content")
    database = main.Database(root / "data" / "app.db")
    monkeypatch.setattr(main, "ROOT", root)
    monkeypatch.setattr(main, "database", database)
    monkeypatch.setattr(main, "ai_service", AIService(database, NoProvider(), api_key="fake"))
    monkeypatch.setattr(main, "library", load_library(root / "content"))

    def reload_test_library():
        main.library = load_library(root / "content")

    monkeypatch.setattr(main, "_reload_library", reload_test_library)
    return TestClient(main.app), database, root


def test_approved_topic_edits_are_atomic_revised_and_keep_id(tmp_path, monkeypatch):
    client, database, root = local_client(tmp_path, monkeypatch)
    before_usage = database.monthly_usage("2026-08")[0]
    editable = client.get("/api/topics/knowledge-distillation/editable")
    assert editable.status_code == 200
    payload = editable.json()["topic"]
    payload["title"] = "Knowledge Distillation — corrected locally"
    saved = client.put("/api/topics/knowledge-distillation", json={"payload": payload, "edit_source": "structured"})
    assert saved.status_code == 200
    assert saved.json()["id"] == "knowledge-distillation"
    assert saved.json()["title"] == payload["title"]
    assert client.get("/api/topics/knowledge-distillation").json()["id"] == "knowledge-distillation"
    assert client.get("/api/topics/knowledge-distillation/question-management").json()["approved_count"] >= 1
    assert client.get("/api/topics/knowledge-distillation/revisions").json()
    assert database.monthly_usage("2026-08")[0] == before_usage
    payload["id"] = "renamed-topic"
    preserved = client.put("/api/topics/knowledge-distillation", json={"payload": payload, "edit_source": "raw_json"})
    assert preserved.status_code == 200 and preserved.json()["id"] == "knowledge-distillation"
    assert (root / "content" / "topics" / "knowledge-distillation.json").is_file()


def test_question_approval_is_local_idempotent_and_recovers_after_retry(tmp_path, monkeypatch):
    client, database, root = local_client(tmp_path, monkeypatch)
    before_usage = database.monthly_usage("2026-08")[0]
    payload = {"topic_id": "gradient-descent", "approval_state": "draft", "generation_metadata": {"review_state": "draft"}, "questions": [
        {"question_category": "intuition", "difficulty": "beginner", "question": "Why does a local approval stay offline?", "direct_answer": "It writes an existing candidate locally.", "expanded_answer": "Approval is a local validation and persistence operation.", "key_points": ["No provider call"], "related_topic_ids": ["missing", "gradient-descent"], "selected": True}
    ]}
    database.create_draft("local-question-draft", "question", "Questions for Gradient Descent", payload)
    first = client.post("/api/ai/drafts/local-question-draft/approve")
    assert first.status_code == 200
    assert len(first.json()["approved"]) == 1
    question_id = first.json()["approved"][0]["id"]
    assert client.get("/api/ai/drafts/local-question-draft").json()["payload"]["approval_state"] == "approved"
    retry = client.post("/api/ai/drafts/local-question-draft/approve")
    assert retry.status_code == 200
    assert retry.json()["approved"] == []
    assert retry.json()["already_approved"][0]["id"] == question_id
    assert len([p for p in (root / "content" / "questions").glob(f"{question_id}.json")]) == 1
    assert database.monthly_usage("2026-08")[0] == before_usage


def test_question_edit_preserves_review_state_and_archiving_is_soft(tmp_path, monkeypatch):
    client, database, _ = local_client(tmp_path, monkeypatch)
    question = next(item for item in client.get("/api/questions").json() if item["id"].startswith("gradient-descent_"))
    database.review(question["id"], "good")
    question["question"] += " (edited locally)"
    saved = client.put(f"/api/questions/{question['id']}", json={"payload": question, "edit_source": "structured"})
    assert saved.status_code == 200
    assert database.progress()[0][0]["question_id"] == question["id"]
    archived = client.post(f"/api/questions/{question['id']}/archive")
    assert archived.status_code == 200
    assert question["id"] not in {item["id"] for item in client.get("/api/questions").json()}
