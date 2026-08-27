from pathlib import Path

from backend import main


ROOT = Path(__file__).resolve().parents[1]


def test_create_topic_dispatch_debug_build_is_cache_busted_and_visible():
    index = (ROOT / "frontend" / "index.html").read_text()
    debug = (ROOT / "frontend" / "topic-generate-debug.js").read_text()
    app = (ROOT / "frontend" / "app.js").read_text()

    assert "/simplified-topic-workflow.js?v=simplified-workflow-v1" in index
    assert "/topic-generate-debug.js?v=topic-generate-debug-v1" in index
    assert "Create topic build: topic-generate-debug-v1" in debug
    assert "button click" in debug
    assert "submit handler entry" in debug
    assert "generated request payload" in debug
    assert "url:'/api/ai/topic-draft',method:'POST'" in debug
    assert "moment before fetch" in app
    assert "timeout callback: aborting request" in app
    assert "fetch response received" in app
    assert "fetch error" in app
    assert "JSON.stringify(details)" in app


def test_topic_draft_route_logs_receive_before_generation(monkeypatch, caplog):
    caplog.set_level("INFO", logger="ultimate_ml.local")
    monkeypatch.setattr(main, "_generate_topic_draft", lambda body: {"id": "fake-draft"})

    response = main.create_topic_draft(main.TopicDraftInput(
        title="Vectors and Vector Spaces",
        category="mathematical_foundations",
        difficulty="beginner",
        depth="ultimate",
    ))

    assert response == {"id": "fake-draft"}
    assert "[TOPIC GENERATE DEBUG] backend received POST /api/ai/topic-draft" in caplog.text
