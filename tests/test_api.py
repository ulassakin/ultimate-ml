import importlib
from fastapi.testclient import TestClient
import backend.main as main

def test_vertical_slice_api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "database", main.Database(tmp_path / "api.db"))
    client = TestClient(main.app)
    summary = client.get("/api/progress/summary").json()
    assert summary["due_today"] == summary["total_questions"]
    topic = client.get("/api/topics/resnet").json()
    assert topic["architecture"]["images"]
    due = client.get("/api/review/due").json()
    assert due[0]["concept_refresher"]["quick_recall"]
    response = client.post(f"/api/review/{due[0]['id']}", json={"rating":"good","answer":"identity path"})
    assert response.status_code == 200
    assert client.get("/api/progress/summary").json()["total_reviews"] == 1

