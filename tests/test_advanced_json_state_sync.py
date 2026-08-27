"""Advanced JSON remains a local, graph-free edit surface."""
from copy import deepcopy

from tests.test_draft_action_reliability import isolated_client, ready_payload


def test_legacy_json_save_strips_graph_fields_and_can_approve(tmp_path, monkeypatch):
    client, database = isolated_client(tmp_path, monkeypatch)
    payload = ready_payload()
    payload["title"] = "DINO Editor State Regression"
    payload["prerequisite_topic_ids"] = ["gradient-descent"]
    payload["metadata_resolution"] = {"durable_edges": {"prerequisites": ["gradient-descent"], "related": []}}
    database.create_draft("dino-editor", "topic", payload["title"], payload)
    applied = deepcopy(payload)
    applied["core_explanation"] = "The teacher produces targets and the student learns to match them without labels."
    saved = client.put("/api/ai/drafts/dino-editor", json={"payload": applied})
    assert saved.status_code == 200
    assert "metadata_resolution" not in saved.json()["payload"]
    assert client.get("/api/ai/drafts/dino-editor/validate").json()["valid"] is True
    assert client.post("/api/ai/drafts/dino-editor/approve").status_code == 200

