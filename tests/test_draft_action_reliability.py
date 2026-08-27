"""Local draft saves and approval remain provider-free after graph removal."""
import shutil

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.service import AIService
from backend.content import load_library


class NoPaidProvider:
    def structured(self, **kwargs):
        raise AssertionError("Local draft save and approval must not call a provider")


def isolated_client(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    shutil.copytree(main.ROOT / "content", root / "content")
    database = main.Database(root / "data" / "app.db")
    monkeypatch.setattr(main, "ROOT", root)
    monkeypatch.setattr(main, "database", database)
    monkeypatch.setattr(main, "ai_service", AIService(database, NoPaidProvider(), api_key="fake"))
    monkeypatch.setattr(main, "library", load_library(root / "content"))
    monkeypatch.setattr(main, "_reload_library", lambda: setattr(main, "library", load_library(root / "content")))
    return TestClient(main.app), database


def ready_payload():
    return {
        "title": "Self-Supervised Learning Action Regression", "category": "representation_learning", "difficulty": "intermediate",
        "concept_type": "broad_concept", "tags": ["self-supervised learning"],
        "one_sentence_summary": "Learning useful representations without manual labels.",
        "quick_recall": "Build a training signal from the data itself.",
        "core_explanation": "Self-supervised objectives create supervision from data structure and learn transferable representations.",
        "quality_review_state": "reviewed", "quality_status": "ready", "quality_report": {"confidence": "high", "blocking_issues_remaining": []},
    }


def test_legacy_graph_metadata_is_ignored_and_local_approval_needs_no_resolver(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = ready_payload() | {
        "prerequisite_topic_ids": ["cnn"], "related_topic_ids": ["resnet"],
        "relationship_justifications": [{"topic_id": "cnn"}], "metadata_resolution": {"durable_edges": {"prerequisites": ["cnn"], "related": ["resnet"]}},
    }
    database.create_draft("ssl", "topic", payload["title"], payload)
    saved = client.put("/api/ai/drafts/ssl", json={"payload": payload})
    assert saved.status_code == 200
    persisted = saved.json()["payload"]
    assert not any(field in persisted for field in ("prerequisite_topic_ids", "related_topic_ids", "relationship_justifications", "metadata_resolution"))
    assert client.get("/api/ai/drafts/ssl/validate").json()["valid"] is True
    approved = client.post("/api/ai/drafts/ssl/approve")
    assert approved.status_code == 200
    assert approved.json()["topic"]["id"] == "self-supervised-learning-action-regression"
    assert database.monthly_usage("2026-08")[0] == 0
