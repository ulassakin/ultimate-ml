from pathlib import Path

from backend import main


ROOT = Path(__file__).resolve().parents[1]


def test_create_topic_action_build_has_immediate_feedback_and_dispatch_logs():
    index = (ROOT / "frontend" / "index.html").read_text()
    app = (ROOT / "frontend" / "app.js").read_text()
    workflow = (ROOT / "frontend" / "simplified-topic-workflow.js").read_text()
    helper = (ROOT / "frontend" / "async-ui.js").read_text()

    assert "/async-ui.js?v=action-reliability-v1" in index
    assert "/simplified-topic-workflow.js?v=action-reliability-v2" in index
    assert "PROVIDER_REQUEST_TIMEOUT_MS=180000" in app
    assert "[${scope}] ${method} ${url} dispatched" in app
    assert "The server may still be processing this request; do not retry yet." in app
    assert "runUiAction({key:'topic-generate'" in workflow
    assert "loadingLabel:'Generating…'" in workflow
    assert "loadingLabel:force?'Re-reviewing…':'Reviewing…'" in workflow
    assert "uiActionsInFlight" in helper and "finally" in helper and "await nextPaint()" in helper


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
