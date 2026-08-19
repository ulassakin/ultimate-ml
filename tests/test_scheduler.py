from datetime import datetime, timezone
import pytest
from backend.scheduler import schedule_review

NOW = datetime(2026, 8, 19, 10, 30, tzinfo=timezone.utc)

def test_again_resets_streak_and_returns_soon():
    result = schedule_review("again", 8, 2.5, 4, NOW)
    assert result.streak == 0
    assert result.interval_days == 0
    assert (result.next_review_at - NOW).total_seconds() == 600

def test_easy_moves_new_question_farther_than_good():
    assert schedule_review("easy", now=NOW).next_review_at > schedule_review("good", now=NOW).next_review_at

def test_invalid_rating_is_rejected():
    with pytest.raises(ValueError):
        schedule_review("perfect", now=NOW)

