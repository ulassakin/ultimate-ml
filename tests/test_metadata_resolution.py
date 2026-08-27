"""The retired resolver API is unavailable; legacy content remains loadable."""
from fastapi.testclient import TestClient

import backend.main as main
from backend.content import load_library


def test_relationship_endpoints_are_removed_and_existing_library_loads():
    client = TestClient(main.app)
    assert client.get("/api/topics/dino/metadata-rebuild-estimate").status_code == 404
    assert client.post("/api/ai/drafts/missing/rebuild-relationships").status_code in {404, 405}
    assert load_library().topics
