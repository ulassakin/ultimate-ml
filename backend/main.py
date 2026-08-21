from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import json
import re
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .content import ROOT, load_library
from .database import Database
from .ai.prompts import QUESTION_INSTRUCTIONS, QUESTION_PROMPT_VERSION, TOPIC_INSTRUCTIONS, TOPIC_PROMPT_VERSION
from .ai.schemas import QuestionDraftBatch, RegeneratedSection, TopicDraft
from .ai.service import AIService, AIUnavailableError, BudgetExceededError, StructuredOutputError
from .ai.structured import strict_response_schema

load_dotenv(ROOT / ".env")
app = FastAPI(title="Ultimate ML", version="0.1.0")
library = load_library()
database = Database(ROOT / "data" / "ultimate_ml.db")
ai_service = AIService(database)


class ReviewInput(BaseModel):
    rating: Literal["again", "hard", "good", "easy"]
    answer: str = ""


class AISettingsInput(BaseModel):
    provider: Literal["openai"] | None = None
    model: str | None = None
    explanation_depth: Literal["standard", "deep", "ultimate"] | None = None
    enabled: bool | None = None
    monthly_budget_usd: float | None = None
    pricing_override: dict | None = None


class TopicDraftInput(BaseModel):
    title: str
    category: str = "classical_ml"
    tags: list[str] = []
    focus: str = ""
    depth: Literal["standard", "deep", "ultimate"] = "ultimate"
    include_mathematics: bool = True
    include_examples: bool = True
    include_misconceptions: bool = True
    suggest_related_topics: bool = True
    allow_duplicate: bool = False


class QuestionDraftInput(BaseModel):
    topic_id: str
    focus: str = ""
    count: int = 5


class DraftPayloadInput(BaseModel):
    payload: dict


class RegenerateSectionInput(BaseModel):
    section: Literal["quick_recall", "big_picture", "why_it_exists", "intuition", "core_explanation", "mechanism", "ml_relevance", "practical_example", "mathematical_foundation", "common_misconceptions", "limitations", "mental_models", "deep_dive"]
    focus: str = ""


def _ai_error(exc):
    if isinstance(exc, BudgetExceededError):
        raise HTTPException(429, str(exc))
    if isinstance(exc, AIUnavailableError):
        raise HTTPException(503, str(exc))
    if isinstance(exc, StructuredOutputError):
        raise HTTPException(422, str(exc))
    raise HTTPException(502, "AI generation failed. Check the local API configuration and try again.")


def _slug(value: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", value.casefold())) or "untitled-topic"


def _title_duplicate(title: str):
    normalized = _slug(title)
    for item in library.topics.values():
        if _slug(item["title"]) == normalized or item["id"] == normalized:
            return {"id": item["id"], "title": item["title"]}
    return None


def _reload_library():
    global library
    library = load_library()


def _approved_topic(payload: dict) -> dict:
    topic = dict(payload)
    topic_id = _slug(topic.get("id") or topic["title"])
    known = set(library.topics)
    topic.update({"id": topic_id, "content_version": 2, "tags": topic.get("tags", []),
        "knowledge_type": topic.get("knowledge_type", ["conceptual"]),
        "prerequisite_topic_ids": [x for x in topic.get("prerequisite_topic_ids", []) if x in known],
        "related_topic_ids": [x for x in topic.get("related_topic_ids", []) if x in known],
        "sources": topic.get("sources", []),
        "generation_metadata": {**topic.get("generation_metadata", {}), "review_state": "approved"}})
    return topic


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def public(item):
    return {key: value for key, value in item.items() if key != "_path"}


@app.get("/api/topics")
def topics(search: str = Query(default="")):
    needle = search.casefold()
    return [public(topic) for topic in library.topics.values()
            if not needle or needle in (topic["title"] + " " + " ".join(topic.get("tags", []))).casefold()]


@app.get("/api/topics/duplicate-check")
def duplicate_check(title: str):
    return {"duplicate": _title_duplicate(title)}


@app.get("/api/topics/{topic_id}")
def topic(topic_id: str):
    if topic_id not in library.topics:
        raise HTTPException(404, "Topic not found")
    item = public(library.topics[topic_id])
    item["related_topic_details"] = [
        {"id": value, "title": library.topics[value]["title"]} for value in item.get("related_topic_ids", [])
    ]
    item["prerequisite_topic_details"] = [
        {"id": value, "title": library.topics[value]["title"]} for value in item.get("prerequisite_topic_ids", [])
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


@app.get("/api/ai/settings")
def ai_settings():
    return ai_service.public_settings()


@app.put("/api/ai/settings")
def update_ai_settings(body: AISettingsInput):
    values = body.model_dump(exclude_none=True)
    if values.get("monthly_budget_usd", 0) < 0:
        raise HTTPException(422, "Monthly budget must be zero or greater.")
    return ai_service.public_settings() | ai_service.update_settings(values).__dict__ | {"api_key_configured": ai_service.key_configured()}


@app.get("/api/ai/usage")
def ai_usage():
    return ai_service.usage_summary()


@app.post("/api/ai/connectivity-test")
def ai_connectivity_test():
    return ai_service.connectivity_test()


@app.post("/api/ai/topic-draft")
def create_topic_draft(body: TopicDraftInput):
    duplicate = _title_duplicate(body.title)
    if duplicate and not body.allow_duplicate:
        raise HTTPException(409, {"message": "A matching topic already exists. Open it or explicitly create a separate draft.", "duplicate": duplicate})
    prompt = {"title": body.title, "primary_category": body.category, "tags": body.tags, "focus": body.focus,
              "depth": body.depth, "include_mathematics": body.include_mathematics, "include_practical_examples": body.include_examples,
              "include_misconceptions": body.include_misconceptions, "suggest_related_topics": body.suggest_related_topics,
              "known_topic_ids": list(library.topics), "future_source_context": "None supplied. Do not invent sources."}
    try:
        draft, result, cost = ai_service.generate(operation_type="topic_draft", instructions=TOPIC_INSTRUCTIONS,
            input_text=json.dumps(prompt), schema_name="ultimate_ml_topic_draft", schema=strict_response_schema(TopicDraft),
            validate=TopicDraft.model_validate, max_output_tokens=5000, metadata={"prompt_version": TOPIC_PROMPT_VERSION})
    except Exception as exc:
        _ai_error(exc)
    payload = draft.model_dump()
    payload["generation_metadata"] = {"generated_by_ai": True, "provider": "openai", "model": ai_service.settings().model,
        "prompt_version": TOPIC_PROMPT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "user_focus": body.focus, "review_state": "draft"}
    draft_id = str(uuid4())
    database.create_draft(draft_id, "topic", payload["title"], payload, {"request_cost_usd": cost})
    return {"id": draft_id, "state": "draft", "payload": payload, "usage": {"input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens, "estimated_cost_usd": cost}, "budget": ai_service.usage_summary()}


@app.post("/api/ai/question-draft")
def create_question_draft(body: QuestionDraftInput):
    topic = library.topics.get(body.topic_id)
    if not topic:
        raise HTTPException(404, "Topic not found")
    request = {"topic_id": topic["id"], "topic_title": topic["title"], "topic_summary": topic["one_sentence_summary"],
               "topic_core_explanation": topic["core_explanation"], "focus": body.focus, "count": min(12, max(1, body.count)),
               "related_topic_ids": topic.get("related_topic_ids", []), "known_question_text": [q["question"] for q in library.questions.values()]}
    try:
        batch, result, cost = ai_service.generate(operation_type="question_draft", instructions=QUESTION_INSTRUCTIONS,
            input_text=json.dumps(request), schema_name="ultimate_ml_question_draft", schema=strict_response_schema(QuestionDraftBatch),
            validate=QuestionDraftBatch.model_validate, max_output_tokens=2600, metadata={"prompt_version": QUESTION_PROMPT_VERSION, "topic_id": body.topic_id})
    except Exception as exc:
        _ai_error(exc)
    payload = {"topic_id": body.topic_id, "questions": [{**item.model_dump(), "selected": True} for item in batch.questions],
        "generation_metadata": {"generated_by_ai": True, "provider": "openai", "model": ai_service.settings().model,
        "prompt_version": QUESTION_PROMPT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "user_focus": body.focus, "review_state": "draft"}}
    draft_id = str(uuid4())
    database.create_draft(draft_id, "question", f"Questions for {topic['title']}", payload, {"request_cost_usd": cost})
    return {"id": draft_id, "state": "draft", "payload": payload, "usage": {"input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens, "estimated_cost_usd": cost}, "budget": ai_service.usage_summary()}


@app.get("/api/ai/drafts/{draft_id}")
def get_draft(draft_id: str):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    return draft


@app.put("/api/ai/drafts/{draft_id}")
def update_draft(draft_id: str, body: DraftPayloadInput):
    if not database.update_draft(draft_id, body.payload):
        raise HTTPException(404, "Draft not found")
    return database.get_draft(draft_id)


@app.post("/api/ai/drafts/{draft_id}/regenerate-section")
def regenerate_section(draft_id: str, body: RegenerateSectionInput):
    draft = database.get_draft(draft_id)
    if not draft or draft["draft_type"] != "topic" or draft["state"] != "draft":
        raise HTTPException(404, "Active topic draft not found")
    current = draft["payload"].get(body.section, "")
    instruction = TOPIC_INSTRUCTIONS + " Regenerate only the requested field; preserve all other draft content and return its replacement value."
    request = {"topic_title": draft["title"], "section": body.section, "current_value": current, "user_focus": body.focus,
               "expected_value_type": "mathematical foundation object" if body.section == "mathematical_foundation" else "text or list"}
    try:
        regenerated, _, _ = ai_service.generate(operation_type="regenerate_section", instructions=instruction,
            input_text=json.dumps(request), schema_name="ultimate_ml_regenerated_section", schema=strict_response_schema(RegeneratedSection),
            validate=RegeneratedSection.model_validate, max_output_tokens=1800, metadata={"section": body.section, "draft_id": draft_id})
    except Exception as exc:
        _ai_error(exc)
    value = regenerated.value.model_dump() if hasattr(regenerated.value, "model_dump") else regenerated.value
    if body.section == "mathematical_foundation" and not isinstance(value, dict):
        raise HTTPException(422, "The regenerated mathematical section had an invalid shape. No draft changes were made.")
    if body.section in {"common_misconceptions", "limitations", "mental_models"} and not isinstance(value, list):
        raise HTTPException(422, "The regenerated list section had an invalid shape. No draft changes were made.")
    if body.section not in {"mathematical_foundation", "common_misconceptions", "limitations", "mental_models"} and not isinstance(value, str):
        raise HTTPException(422, "The regenerated text section had an invalid shape. No draft changes were made.")
    draft["payload"][body.section] = value
    database.update_draft(draft_id, draft["payload"])
    return database.get_draft(draft_id)


@app.post("/api/ai/drafts/{draft_id}/discard")
def discard_draft(draft_id: str):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    database.update_draft(draft_id, draft["payload"], "discarded")
    return database.get_draft(draft_id)


@app.post("/api/ai/drafts/{draft_id}/approve")
def approve_draft(draft_id: str):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    if draft["state"] != "draft":
        raise HTTPException(409, "Only an active draft can be approved")
    if draft["draft_type"] == "topic":
        topic = _approved_topic(draft["payload"])
        target = ROOT / "content" / "topics" / f"{topic['id']}.json"
        if target.exists():
            raise HTTPException(409, "A topic with this ID already exists. Edit the draft title or use the existing topic.")
        _write_json(target, topic)
        try:
            _reload_library()
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(422, f"Draft cannot be approved: {exc}") from exc
        database.update_draft(draft_id, topic, "approved")
        return {"state": "approved", "topic": public(library.topics[topic["id"]])}
    payload = draft["payload"]
    topic_id = payload.get("topic_id")
    selected = [item for item in payload.get("questions", []) if item.get("selected")]
    if not topic_id or topic_id not in library.topics or not selected:
        raise HTTPException(422, "Select at least one question for an existing topic before approval")
    saved = []
    existing_text = {_slug(item["question"]): item["id"] for item in library.questions.values()}
    for item in selected:
        if _slug(item["question"]) in existing_text:
            continue
        question_id = f"{topic_id}_q{len(library.questions) + len(saved) + 1:03d}"
        question = {"content_version": 2, "id": question_id, "topic_ids": [topic_id], "question_type": "conceptual",
            **{key: value for key, value in item.items() if key != "selected"}, "concept_refresher_topic_id": topic_id,
            "generation_metadata": {**payload["generation_metadata"], "review_state": "approved"}}
        _write_json(ROOT / "content" / "questions" / f"{question_id}.json", question)
        saved.append(question_id)
    if not saved:
        raise HTTPException(409, "All selected questions duplicate the current question bank")
    _reload_library()
    database.update_draft(draft_id, payload, "approved")
    return {"state": "approved", "question_ids": saved}


app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
