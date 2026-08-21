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
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ai_usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, calendar_month TEXT NOT NULL,
                    provider TEXT NOT NULL, model TEXT NOT NULL, operation_type TEXT NOT NULL,
                    topic_id TEXT, question_id TEXT, input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0, cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0, status TEXT NOT NULL,
                    error_type TEXT, metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS ai_usage_month_idx ON ai_usage_events(calendar_month, status);
                CREATE TABLE IF NOT EXISTS ai_generation_drafts (
                    id TEXT PRIMARY KEY, draft_type TEXT NOT NULL, state TEXT NOT NULL,
                    title TEXT NOT NULL, payload_json TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
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

    def get_setting(self, key: str):
        import json
        with self.connect() as db:
            row = db.execute("SELECT value_json FROM app_settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value_json"]) if row else None

    def set_setting(self, key: str, value):
        import json
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("""INSERT INTO app_settings(key,value_json,updated_at) VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at""",
                (key, json.dumps(value), now))

    def create_usage_event(self, *, provider, model, operation_type, estimated_cost_usd, status="reserved",
                           topic_id=None, question_id=None, metadata=None, now=None):
        import json
        now = now or datetime.now(timezone.utc)
        with self.connect() as db:
            cursor = db.execute("""INSERT INTO ai_usage_events
                (timestamp,calendar_month,provider,model,operation_type,topic_id,question_id,estimated_cost_usd,status,metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (now.isoformat(), now.strftime("%Y-%m"), provider, model, operation_type, topic_id, question_id,
                 estimated_cost_usd, status, json.dumps(metadata or {})))
        return cursor.lastrowid

    def finalize_usage_event(self, event_id, *, input_tokens=0, output_tokens=0, cached_input_tokens=0,
                             estimated_cost_usd=0, status="success", error_type=None):
        with self.connect() as db:
            db.execute("""UPDATE ai_usage_events SET input_tokens=?,output_tokens=?,cached_input_tokens=?,
                estimated_cost_usd=?,status=?,error_type=? WHERE id=?""",
                (input_tokens, output_tokens, cached_input_tokens, estimated_cost_usd, status, error_type, event_id))

    def monthly_usage(self, calendar_month: str):
        with self.connect() as db:
            row = db.execute("""SELECT COALESCE(SUM(estimated_cost_usd), 0) AS total FROM ai_usage_events
                WHERE calendar_month = ? AND status IN ('reserved','success')""", (calendar_month,)).fetchone()
            groups = db.execute("""SELECT operation_type,model,COALESCE(SUM(estimated_cost_usd),0) AS estimated_cost_usd,
                SUM(input_tokens) AS input_tokens,SUM(output_tokens) AS output_tokens,SUM(cached_input_tokens) AS cached_input_tokens
                FROM ai_usage_events WHERE calendar_month=? GROUP BY operation_type,model ORDER BY estimated_cost_usd DESC""",
                (calendar_month,)).fetchall()
        return float(row["total"]), [dict(group) for group in groups]

    def create_draft(self, draft_id, draft_type, title, payload, metadata=None, now=None):
        import json
        now = now or datetime.now(timezone.utc)
        with self.connect() as db:
            db.execute("""INSERT INTO ai_generation_drafts(id,draft_type,state,title,payload_json,metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?)""", (draft_id, draft_type, "draft", title, json.dumps(payload),
                json.dumps(metadata or {}), now.isoformat(), now.isoformat()))

    def get_draft(self, draft_id):
        import json
        with self.connect() as db:
            row = db.execute("SELECT * FROM ai_generation_drafts WHERE id=?", (draft_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def update_draft(self, draft_id, payload, state=None):
        import json
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            cursor = db.execute("UPDATE ai_generation_drafts SET payload_json=?,state=COALESCE(?,state),updated_at=? WHERE id=?",
                (json.dumps(payload), state, now, draft_id))
        return cursor.rowcount > 0
