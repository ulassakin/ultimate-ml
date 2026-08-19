from datetime import datetime, timezone
from backend.database import Database

NOW = datetime(2026, 8, 19, 10, 30, tzinfo=timezone.utc)

def test_review_and_history_are_persisted(tmp_path):
    db = Database(tmp_path / "test.db")
    saved = db.review("resnet_q001", "good", "my answer", NOW)
    progress, history = db.progress()
    assert saved["interval_days"] == 1
    assert progress[0]["review_count"] == 1
    assert history[0]["answer"] == "my answer"
    db.review("resnet_q001", "hard", "second", NOW)
    progress, history = db.progress()
    assert progress[0]["review_count"] == 2
    assert len(history) == 2

