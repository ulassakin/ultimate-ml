"""Small static coverage for browser-only async reliability (no JS test runner is bundled)."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_draft_actions_have_timeout_finally_single_flight_and_visible_errors():
    app = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    actions = (ROOT / "frontend" / "draft-action-reliability.js").read_text(encoding="utf-8")
    assert "AbortController" in app and "REQUEST_TIMEOUT_MS" in app and "Request timed out after" in app
    assert "HTTP ${response.status}" in app and "Malformed response" in app and "Network request failed" in app
    assert "if(draftEditorState.active)" in actions
    assert "finally{draftEditorState.active=false" in actions
    assert "mutationButtons(true)" in actions and "mutationButtons(false)" in actions
    assert "Save edits" in actions and "Rebuild relationships" in actions and "Validate and approve" in actions
    assert "persistedDraft" in actions and "adoptPersistedDraft" in actions


def test_advanced_json_uses_one_canonical_payload_and_guards_stale_approval():
    actions = (ROOT / "frontend" / "draft-action-reliability.js").read_text(encoding="utf-8")
    assert "const draftEditorState =" in actions
    assert "draftEditorState.payload=deepCopy(payload);draftEditorState.dirty=true" in actions
    assert "Advanced JSON applied to current draft state" in actions
    assert "schemaError(payload)" in actions and "Invalid JSON:" in actions
    assert "renderCanonicalPayload();bindStructuredEditors();" in actions
    assert "const body={payload:deepCopy(draftEditorState.payload)};" in actions
    assert "request('/ai/drafts/'+draftEditorState.draftId,'PUT',body)" in actions
    assert "if(draftEditorState.dirty)await persistCanonical();" in actions
    assert "refreshAuditStaleness" in actions
    assert "sameRelations(edges.prerequisites,payload.prerequisite_topic_ids)" in actions
    assert "ensureMathPrerequisiteEditor" in actions and "#math-prerequisites" in actions
