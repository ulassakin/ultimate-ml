import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .scheduler import schedule_review


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.initialize()

    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self):
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS question_progress (
                    question_id TEXT PRIMARY KEY, last_reviewed_at TEXT,
                    next_review_at TEXT NOT NULL, review_count INTEGER NOT NULL DEFAULT 0,
                    interval_days INTEGER NOT NULL DEFAULT 0, ease REAL NOT NULL DEFAULT 2.5,
                    streak INTEGER NOT NULL DEFAULT 0, last_rating TEXT
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, question_id TEXT NOT NULL,
                    rating TEXT NOT NULL, answer TEXT NOT NULL DEFAULT '', reviewed_at TEXT NOT NULL,
                    next_review_at TEXT NOT NULL
                );
            """)

    def review(self, question_id: str, rating: str, answer: str = "", now: datetime | None = None):
        now = now or datetime.now(timezone.utc)
        with self.connect() as db:
            previous = db.execute("SELECT * FROM question_progress WHERE question_id = ?", (question_id,)).fetchone()
            result = schedule_review(rating, previous["interval_days"] if previous else 0,
                                     previous["ease"] if previous else 2.5,
                                     previous["streak"] if previous else 0, now)
            next_at = result.next_review_at.isoformat()
            db.execute("""INSERT INTO question_progress
                (question_id,last_reviewed_at,next_review_at,review_count,interval_days,ease,streak,last_rating)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(question_id) DO UPDATE SET
                last_reviewed_at=excluded.last_reviewed_at,next_review_at=excluded.next_review_at,
                review_count=question_progress.review_count+1,interval_days=excluded.interval_days,
                ease=excluded.ease,streak=excluded.streak,last_rating=excluded.last_rating""",
                (question_id, now.isoformat(), next_at, 1, result.interval_days, result.ease, result.streak, rating))
            db.execute("INSERT INTO reviews(question_id,rating,answer,reviewed_at,next_review_at) VALUES(?,?,?,?,?)",
                       (question_id, rating, answer, now.isoformat(), next_at))
        return {"question_id": question_id, "rating": rating, "next_review_at": next_at,
                "interval_days": result.interval_days, "streak": result.streak}

    def progress(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM question_progress").fetchall()
            recent = db.execute("SELECT * FROM reviews ORDER BY reviewed_at DESC LIMIT 8").fetchall()
        return [dict(row) for row in rows], [dict(row) for row in recent]

