import importlib
from fastapi.testclient import TestClient
import backend.main as main
from backend.ai.provider import ProviderResult
from backend.ai.service import AIService
from backend.ai.schemas import TopicDraft

def test_vertical_slice_api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "database", main.Database(tmp_path / "api.db"))
    client = TestClient(main.app)
    summary = client.get("/api/progress/summary").json()
    assert summary["due_today"] == summary["total_questions"]
    topic = client.get("/api/topics/resnet").json()
    assert topic["architecture"]["images"]
    pca = client.get("/api/topics/pca").json()
    assert pca["prerequisite_topic_details"] == [{"id": "covariance", "title": "Covariance"}]
    due = client.get("/api/review/due").json()
    assert due[0]["concept_refresher"]["quick_recall"]
    response = client.post(f"/api/review/{due[0]['id']}", json={"rating":"good","answer":"identity path"})
    assert response.status_code == 200
    assert client.get("/api/progress/summary").json()["total_reviews"] == 1


def test_ai_settings_work_without_exposing_api_key(tmp_path, monkeypatch):
    fake_db = main.Database(tmp_path / "api.db")
    monkeypatch.setattr(main, "database", fake_db)
    monkeypatch.setattr(main, "ai_service", AIService(fake_db, api_key="not-visible"))
    client = TestClient(main.app)
    settings = client.get("/api/ai/settings").json()
    assert settings["api_key_configured"] is True
    assert "api_key" not in settings
    updated = client.put("/api/ai/settings", json={"monthly_budget_usd": 3.0}).json()
    assert updated["monthly_budget_usd"] == 3.0


def test_ai_topic_draft_uses_fake_provider_and_stays_a_draft(tmp_path, monkeypatch):
    class FakeProvider:
        def structured(self, **kwargs):
            return ProviderResult({"title":"Gaussian Mixture Models","category":"classical_ml","one_sentence_summary":"A mixture model.","quick_recall":"Soft assignments.","core_explanation":"Weighted Gaussian components."}, 20, 30)
    fake_db = main.Database(tmp_path / "api.db")
    monkeypatch.setattr(main, "database", fake_db)
    monkeypatch.setattr(main, "ai_service", AIService(fake_db, FakeProvider(), api_key="fake"))
    client = TestClient(main.app)
    response = client.post("/api/ai/topic-draft", json={"title":"Gaussian Mixture Models", "focus":"EM and covariance"})
    assert response.status_code == 200
    draft = response.json()
    assert draft["state"] == "draft"
    assert draft["payload"]["generation_metadata"]["review_state"] == "draft"
    assert fake_db.get_draft(draft["id"])["state"] == "draft"


def test_ai_draft_returns_setup_message_without_a_key(tmp_path, monkeypatch):
    fake_db = main.Database(tmp_path / "api.db")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "database", fake_db)
    monkeypatch.setattr(main, "ai_service", AIService(fake_db))
    response = TestClient(main.app).post("/api/ai/topic-draft", json={"title":"New Topic"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_connectivity_endpoint_uses_fake_backend_only_provider(tmp_path, monkeypatch):
    class FakeProvider:
        def connectivity(self, **kwargs): return ProviderResult({"ok": True}, 4, 2)
    fake_db = main.Database(tmp_path / "api.db")
    monkeypatch.setattr(main, "database", fake_db)
    monkeypatch.setattr(main, "ai_service", AIService(fake_db, FakeProvider(), api_key="fake"))
    result = TestClient(main.app).post("/api/ai/connectivity-test").json()
    assert result["key_valid"] is True
    assert result["billing_model_access"] is True
