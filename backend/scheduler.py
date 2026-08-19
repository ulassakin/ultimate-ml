from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

RATINGS = {"again", "hard", "good", "easy"}


@dataclass(frozen=True)
class Schedule:
    interval_days: int
    ease: float
    streak: int
    next_review_at: datetime


def schedule_review(rating: str, previous_interval: int = 0, ease: float = 2.5, streak: int = 0,
                    now: datetime | None = None) -> Schedule:
    if rating not in RATINGS:
        raise ValueError(f"Unknown rating: {rating}")
    now = now or datetime.now(timezone.utc)
    if rating == "again":
        interval, next_streak, next_ease = 0, 0, max(1.3, ease - 0.2)
        due = now + timedelta(minutes=10)
    elif rating == "hard":
        interval, next_streak, next_ease = max(1, round(max(1, previous_interval) * 1.2)), streak + 1, max(1.3, ease - 0.15)
        due = now + timedelta(days=interval)
    elif rating == "good":
        interval = 1 if streak == 0 else (3 if streak == 1 else max(4, round(previous_interval * ease)))
        next_streak, next_ease, due = streak + 1, ease, now + timedelta(days=interval)
    else:
        interval = 4 if streak == 0 else max(7, round(max(1, previous_interval) * (ease + 0.3)))
        next_streak, next_ease, due = streak + 1, ease + 0.05, now + timedelta(days=interval)
    return Schedule(interval, next_ease, next_streak, due)

