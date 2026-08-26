"""Draft persistence regression for the SSL save → rebuild → validate → approve path."""
import shutil

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.metadata_resolution import MetadataResolutionService
from backend.ai.service import AIService
from backend.ai.topic_quality import apply_relationship_resolution
from backend.content import load_library


class NoPaidProvider:
    def structured(self, **kwargs):
        raise AssertionError("This local save/rebuild/approval regression must not call a provider")


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


def ssl_payload(database, catalog):
    payload = {"title": "Self-Supervised Learning Action Regression", "category": "representation_learning", "difficulty": "intermediate",
        "concept_type": "broad_concept", "tags": ["self-supervised learning"],
        "one_sentence_summary": "Learning useful representations without manual labels.",
        "quick_recall": "Build a training signal from the data itself.",
        "core_explanation": "Self-supervised objectives create supervision from data structure, then learn transferable representations.",
        "prerequisite_topic_ids": [], "related_topic_ids": [], "relationship_justifications": [],
        "quality_status": "ready", "quality_report": {"confidence": "high", "blocking_issues_remaining": []}}
    candidates = [{"topic_id": key, "title": item["title"]} for key, item in catalog.items()]
    resolution = {"resolver_version": "relationship-resolver-v1", "prerequisites": [], "related": [
        {"topic_id": "simclr", "confidence": "high", "reason": "SimCLR is a canonical named contrastive method for this learning paradigm."},
        {"topic_id": "contrastive-language-image-pretraining-clip", "confidence": "high", "reason": "CLIP is a central multimodal extension of contrastive representation learning."}], "rejected_candidates": []}
    payload, _, _ = apply_relationship_resolution(payload, resolution, candidates=candidates, assigned_topic_id="self-supervised-learning-action-regression")
    service = MetadataResolutionService(database)
    retrieved = service.retrieve(payload, catalog)
    key, topic_hash, hashes = service.cache_key(payload, retrieved, "relationship-resolver-v1")
    database.set_metadata_resolution_cache(key, topic_hash, hashes, "relationship-resolver-v1", resolution)
    payload["metadata_resolution"].update({"cache_key": key, "topic_hash": topic_hash, "candidate_hashes": hashes,
        "cached": True, "resolver_prompt_version": "relationship-resolver-v1"})
    return payload


def test_ssl_save_rebuild_validate_and_approve_are_canonical_and_local(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = ssl_payload(database, main.library.topics)
    database.create_draft("ssl", "topic", payload["title"], payload)
    assert client.get("/api/ai/drafts/ssl").status_code == 200

    # Simulate selector ordering from the UI; the audit has the opposite order.
    saved_payload = {**payload, "related_topic_ids": ["simclr", "contrastive-language-image-pretraining-clip"]}
    saved = client.put("/api/ai/drafts/ssl", json={"payload": saved_payload})
    assert saved.status_code == 200
    persisted = saved.json()["payload"]
    assert persisted["related_topic_ids"] == ["contrastive-language-image-pretraining-clip", "simclr"]
    assert persisted["metadata_resolution"]["durable_edges"]["related"] == persisted["related_topic_ids"]

    rebuilt = client.post("/api/ai/drafts/ssl/rebuild-relationships")
    assert rebuilt.status_code == 200 and rebuilt.json()["usage"]["cached"] is True
    persisted = rebuilt.json()["draft"]["payload"]
    assert persisted["related_topic_ids"] == persisted["metadata_resolution"]["durable_edges"]["related"]
    validation = client.get("/api/ai/drafts/ssl/validate")
    assert validation.status_code == 200 and validation.json()["valid"] is True
    approved = client.post("/api/ai/drafts/ssl/approve")
    assert approved.status_code == 200 and approved.json()["topic"]["id"] == "self-supervised-learning-action-regression"
    assert database.monthly_usage("2026-08")[0] == 0


def test_rebuild_replaces_stale_saved_relationships_with_one_atomic_resolver_snapshot(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = ssl_payload(database, main.library.topics)
    database.create_draft("ssl", "topic", payload["title"], payload)
    stale = {**payload, "related_topic_ids": ["simclr"], "relationship_justifications": [
        item for item in payload["relationship_justifications"] if item["topic_id"] == "simclr"]}
    assert client.put("/api/ai/drafts/ssl", json={"payload": stale}).status_code == 200
    rebuilt = client.post("/api/ai/drafts/ssl/rebuild-relationships")
    assert rebuilt.status_code == 200
    persisted = rebuilt.json()["draft"]["payload"]
    expected = ["contrastive-language-image-pretraining-clip", "simclr"]
    assert persisted["related_topic_ids"] == expected
    assert persisted["metadata_resolution"]["durable_edges"]["related"] == expected
    assert {item["topic_id"] for item in persisted["relationship_justifications"] if item["relationship"] == "related"} == set(expected)


def test_rebuild_server_failure_leaves_draft_unchanged(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = ssl_payload(database, main.library.topics)
    database.create_draft("ssl", "topic", payload["title"], payload)
    before = database.get_draft("ssl")["payload"]
    monkeypatch.setattr(main, "_resolve_topic_metadata", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("resolver unavailable")))
    failing_client = TestClient(main.app, raise_server_exceptions=False)
    response = failing_client.post("/api/ai/drafts/ssl/rebuild-relationships")
    assert response.status_code == 500
    assert database.get_draft("ssl")["payload"] == before


def test_draft_actions_return_actionable_422_errors_without_persisting_partial_changes(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    database.create_draft("bad", "topic", "Bad", {"title": "Bad", "category": "representation_learning",
        "one_sentence_summary": "x", "quick_recall": "x", "core_explanation": "x"})
    validation = client.get("/api/ai/drafts/bad/validate")
    assert validation.status_code == 200 and validation.json()["valid"] is False
    approval = client.post("/api/ai/drafts/bad/approve")
    assert approval.status_code == 422
    detail = approval.json()["detail"]
    assert detail["message"] == "Fix draft validation errors before approval."
