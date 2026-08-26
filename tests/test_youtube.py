import json

from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.provider import ProviderResult
from backend.ai.service import AIService
from backend.youtube.schemas import TranscriptDocument, TranscriptSegment, TranscriptSource
from backend.youtube.service import YoutubeService


class FakeYoutubeProvider:
    def __init__(self):
        self.calls = []

    def structured(self, **kwargs):
        self.calls.append(kwargs["schema_name"])
        if kwargs["schema_name"] == "ultimate_ml_youtube_concepts":
            return ProviderResult({"learning_arc": "Optimization is introduced as an iterative learning process.", "concepts": [
                {"canonical_name": "Gradient Descent", "importance": "core", "source_evidence_summary": "The speaker explains moving opposite the gradient.", "ml_learning_value": "It is the standard optimization intuition.", "timestamp_seconds": [12.0]},
                {"canonical_name": "Learning Rate", "importance": "supporting", "source_evidence_summary": "Step size controls each update.", "ml_learning_value": "It controls optimization stability.", "timestamp_seconds": [25.0]},
            ]}, 30, 40)
        if kwargs["schema_name"] == "ultimate_ml_topic_draft":
            return ProviderResult({"title": "Learning Rate", "category": "ml_fundamentals", "difficulty": "intermediate",
                "one_sentence_summary": "A step-size control for optimization.", "quick_recall": "It scales a parameter update.",
                "core_explanation": "It balances progress and stability during iterative optimization."}, 25, 30)
        if kwargs["schema_name"] == "ultimate_ml_topic_quality_review":
            candidate=json.loads(kwargs["input_text"])["candidate"]
            from backend.ai.schemas import TopicDraft
            corrected={key:value for key,value in candidate.items() if key in TopicDraft.model_fields}
            return ProviderResult({"corrected_topic": corrected, "quality_report": {"confidence":"high"}}, 10, 15)
        if kwargs["schema_name"] == "ultimate_ml_relationship_resolution":
            request=json.loads(kwargs["input_text"])
            assert "topic_catalog" not in request
            return ProviderResult({"prerequisites":[],"related":[],"rejected_candidates":[]}, 5, 5)
        if kwargs["schema_name"] == "ultimate_ml_question_draft":
            return ProviderResult({"questions": [{"question_category": "intuition", "difficulty": "intermediate",
                "question": "Why can a learning rate that is too large destabilize gradient descent?", "direct_answer": "It can overshoot useful descent directions.",
                "expanded_answer": "Large steps can pass across a valley instead of making stable progress."}]}, 20, 25)
        raise AssertionError(f"Unexpected model operation: {kwargs['schema_name']}")


def setup_youtube_client(tmp_path, monkeypatch, provider=None):
    fake_db = main.Database(tmp_path / "api.db")
    monkeypatch.setattr(main, "database", fake_db)
    monkeypatch.setattr(main, "ai_service", AIService(fake_db, provider or FakeYoutubeProvider(), api_key="fake"))
    monkeypatch.setattr(main, "youtube_service", YoutubeService(tmp_path / "youtube_cache"))
    return TestClient(main.app), fake_db


def test_pasted_transcript_is_cached_locally_and_analyzed_with_fake_provider(tmp_path, monkeypatch):
    client, fake_db = setup_youtube_client(tmp_path, monkeypatch)
    created = client.post("/api/youtube/imports", json={"title": "Optimization lecture", "pasted_transcript": "Gradient descent moves opposite the gradient.\nThe learning rate controls step size."})
    assert created.status_code == 200
    imported = created.json()
    assert imported["source"]["kind"] == "pasted"
    assert imported["transcript"]["cached_locally"] is True
    assert "segments" not in imported
    analysis = client.post(f"/api/youtube/imports/{imported['id']}/analyze")
    assert analysis.status_code == 200
    concepts = analysis.json()["analysis"]["concepts"]
    assert concepts[0]["canonical_name"] == "Gradient Descent"
    assert concepts[0]["existing_topic_match"]["id"] == "gradient-descent"
    assert concepts[0]["allowed_actions"] == ["enrich", "ignore"]
    assert analysis.json()["usage"]["events"][0]["operation_type"] == "youtube_concept_extraction"
    assert fake_db.get_youtube_import(imported["id"])["transcript_cache_path"].endswith(".json")


def test_youtube_url_uses_isolated_provider_and_manual_mode_remains_available(tmp_path, monkeypatch):
    client, _ = setup_youtube_client(tmp_path, monkeypatch)

    def fake_retrieve(self, url, **kwargs):
        return TranscriptDocument(source=TranscriptSource(kind="youtube", video_url=url, video_id="abc123", title="Remote title", transcript_attribution="YouTube transcript retrieved locally"), segments=[TranscriptSegment(text="A source transcript.", start_seconds=0)])

    monkeypatch.setattr(main.YouTubeTranscriptProvider, "retrieve", fake_retrieve)
    response = client.post("/api/youtube/imports", json={"video_url": "https://www.youtube.com/watch?v=abc123"})
    assert response.status_code == 200
    assert response.json()["source"]["video_id"] == "abc123"


def test_youtube_import_requires_one_explicit_transcript_source(tmp_path, monkeypatch):
    client, _ = setup_youtube_client(tmp_path, monkeypatch)
    assert client.post("/api/youtube/imports", json={}).status_code == 422
    assert client.post("/api/youtube/imports", json={"video_url": "https://youtu.be/x", "pasted_transcript": "also here"}).status_code == 422


def test_video_expansion_and_questions_are_drafts_with_video_usage(tmp_path, monkeypatch):
    client, _ = setup_youtube_client(tmp_path, monkeypatch)
    imported = client.post("/api/youtube/imports", json={"pasted_transcript": "Gradient descent and learning rate are introduced."}).json()
    client.post(f"/api/youtube/imports/{imported['id']}/analyze")
    expanded = client.post(f"/api/youtube/imports/{imported['id']}/concepts/1/expand", json={"action": "create"})
    assert expanded.status_code == 200
    assert expanded.json()["payload"]["generation_metadata"]["youtube_import_id"] == imported["id"]
    questions = client.post(f"/api/youtube/imports/{imported['id']}/question-draft", json={"topic_id": "gradient-descent"})
    assert questions.status_code == 200
    assert questions.json()["payload"]["generation_metadata"]["youtube_import_id"] == imported["id"]
    usage = client.get(f"/api/youtube/imports/{imported['id']}").json()["usage"]["events"]
    assert {event["operation_type"] for event in usage} == {"youtube_concept_extraction", "youtube_topic_expansion", "youtube_topic_quality_review", "metadata_relationship_resolution", "youtube_question_generation"}


def test_existing_video_draft_survives_queue_projection_without_payload_change(tmp_path, monkeypatch):
    client, fake_db = setup_youtube_client(tmp_path, monkeypatch)
    imported = client.post("/api/youtube/imports", json={"pasted_transcript": "A transcript."}).json()
    original = {"title": "Contrastive Learning", "difficulty": "intermediate", "category": "ml_fundamentals",
                "one_sentence_summary": "x", "quick_recall": "x", "core_explanation": "x"}
    fake_db.create_draft("old", "topic", "Contrastive Learning", original,
        {"youtube_import_id": imported["id"], "youtube_concept": "Contrastive Learning"})
    queue = client.get(f"/api/youtube/imports/{imported['id']}/draft-queue").json()["queue"]
    assert queue[0]["draft_id"] == "old"
    assert queue[0]["status"] == "ready"
    assert fake_db.get_draft("old")["payload"] == original


def test_batch_preflight_queue_generation_and_duplicate_reuse(tmp_path, monkeypatch):
    provider = FakeYoutubeProvider()
    client, _ = setup_youtube_client(tmp_path, monkeypatch, provider)
    imported = client.post("/api/youtube/imports", json={"pasted_transcript": "Gradient descent and learning rate are introduced."}).json()
    client.post(f"/api/youtube/imports/{imported['id']}/analyze")
    selections = {"selections": [{"concept_index": 0, "action": "enrich"}, {"concept_index": 1, "action": "create"}]}
    preflight = client.post(f"/api/youtube/imports/{imported['id']}/draft-queue/preflight", json=selections).json()
    assert preflight["selected_count"] == 2 and preflight["billable_count"] == 2 and preflight["safe_to_start"]
    queued = client.post(f"/api/youtube/imports/{imported['id']}/draft-queue", json=selections).json()["queue"]
    assert [item["status"] for item in queued] == ["pending", "pending"]
    client.post(f"/api/youtube/imports/{imported['id']}/draft-queue/generate-next")
    queue = client.get(f"/api/youtube/imports/{imported['id']}/draft-queue").json()["queue"]
    ready = next(item for item in queue if item["status"] == "ready")
    before = provider.calls.count("ultimate_ml_topic_draft")
    reused = client.post(f"/api/youtube/imports/{imported['id']}/concepts/0/expand", json={"action": "enrich"}).json()
    assert reused["reused"] is True and reused["id"] == ready["draft_id"]
    assert provider.calls.count("ultimate_ml_topic_draft") == before


def test_reanalysis_v2_preserves_drafts_and_named_methods_remain_distinct(tmp_path, monkeypatch):
    class NamedProvider(FakeYoutubeProvider):
        def structured(self, **kwargs):
            if kwargs["schema_name"] == "ultimate_ml_youtube_concepts":
                self.calls.append(kwargs["schema_name"])
                return ProviderResult({"learning_arc": "Contrastive methods are compared.", "concepts": [
                    {"canonical_name": "Contrastive Learning", "importance": "core", "concept_type": "broad_concept", "source_evidence_summary": "Explains positive and negative pairs.", "ml_learning_value": "Broad objective.", "timestamp_seconds": [1]},
                    {"canonical_name": "SimCLR", "importance": "supporting", "concept_type": "named_method", "parent_concepts": ["Contrastive Learning"], "source_evidence_summary": "Compares its augmentation-based objective.", "ml_learning_value": "Named historical method.", "timestamp_seconds": [4]},
                    {"canonical_name": "CLIP", "importance": "supporting", "concept_type": "named_method", "parent_concepts": ["Contrastive Learning"], "source_evidence_summary": "Explains image-text contrastive training.", "ml_learning_value": "Named multimodal method.", "timestamp_seconds": [9]},
                ]}, 20, 20)
            return super().structured(**kwargs)
    provider = NamedProvider()
    client, fake_db = setup_youtube_client(tmp_path, monkeypatch, provider)
    imported = client.post("/api/youtube/imports", json={"pasted_transcript": "Methods are discussed."}).json()
    client.post(f"/api/youtube/imports/{imported['id']}/analyze")
    fake_db.create_draft("contrastive", "topic", "Contrastive Learning", {"title": "Contrastive Learning", "unchanged": True},
        {"youtube_import_id": imported["id"], "youtube_concept": "Contrastive Learning"})
    before = fake_db.get_draft("contrastive")["payload"]
    result = client.post(f"/api/youtube/imports/{imported['id']}/analyze").json()
    names = {concept["canonical_name"] for concept in result["analysis"]["concepts"]}
    assert {"Contrastive Learning", "SimCLR", "CLIP"} <= names
    contrastive = next(concept for concept in result["analysis"]["concepts"] if concept["canonical_name"] == "Contrastive Learning")
    assert contrastive["draft"]["id"] == "contrastive"
    assert fake_db.get_draft("contrastive")["payload"] == before
    assert result["analysis"]["prompt_version"] == "youtube-concept-extraction-v2"
