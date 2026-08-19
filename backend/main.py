from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .content import ROOT, load_library
from .database import Database

app = FastAPI(title="Ultimate ML", version="0.1.0")
library = load_library()
database = Database(ROOT / "data" / "ultimate_ml.db")


class ReviewInput(BaseModel):
    rating: Literal["again", "hard", "good", "easy"]
    answer: str = ""


def public(item):
    return {key: value for key, value in item.items() if key != "_path"}


@app.get("/api/topics")
def topics(search: str = Query(default="")):
    needle = search.casefold()
    return [public(topic) for topic in library.topics.values()
            if not needle or needle in (topic["title"] + " " + " ".join(topic.get("tags", []))).casefold()]


@app.get("/api/topics/{topic_id}")
def topic(topic_id: str):
    if topic_id not in library.topics:
        raise HTTPException(404, "Topic not found")
    item = public(library.topics[topic_id])
    item["related_topic_details"] = [
        {"id": value, "title": library.topics[value]["title"]} for value in item.get("related_topics", [])
    ]
    return item


@app.get("/api/questions")
def questions():
    return [public(question) for question in library.questions.values()]


@app.get("/api/review/due")
def due_questions():
    progress, _ = database.progress()
    states = {row["question_id"]: row for row in progress}
    now = datetime.now(timezone.utc).isoformat()
    due = [question for question in library.questions.values()
           if question["id"] not in states or states[question["id"]]["next_review_at"] <= now]
    return [{**public(question), "concept_refresher": public(library.topics[question["concept_refresher_topic_id"]])}
            for question in due]


@app.post("/api/review/{question_id}")
def save_review(question_id: str, body: ReviewInput):
    if question_id not in library.questions:
        raise HTTPException(404, "Question not found")
    return database.review(question_id, body.rating, body.answer)


@app.get("/api/progress/summary")
def summary():
    progress, recent = database.progress()
    now = datetime.now(timezone.utc).isoformat()
    states = {row["question_id"]: row for row in progress}
    weak_counts = {}
    for question in library.questions.values():
        state = states.get(question["id"])
        if state and state["last_rating"] in ("again", "hard"):
            for topic_id in question["topic_ids"]:
                weak_counts[topic_id] = weak_counts.get(topic_id, 0) + 1
    return {"total_topics": len(library.topics), "total_questions": len(library.questions),
            "due_today": sum(1 for q in library.questions.values() if q["id"] not in states or states[q["id"]]["next_review_at"] <= now),
            "reviewed_questions": len(progress), "total_reviews": sum(row["review_count"] for row in progress),
            "weak_topics": [{"id": key, "title": library.topics[key]["title"], "count": count}
                            for key, count in sorted(weak_counts.items(), key=lambda item: -item[1])],
            "recent_reviews": recent}


app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")

