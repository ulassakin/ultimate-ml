import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"


class ContentError(ValueError):
    pass


@dataclass
class Library:
    topics: dict[str, dict[str, Any]]
    questions: dict[str, dict[str, Any]]


def _documents(content_root: Path, folder: str) -> list[dict[str, Any]]:
    documents = []
    for path in sorted((content_root / folder).glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ContentError(f"{path}: {exc}") from exc
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        document["_path"] = str(display_path)
        documents.append(document)
    return documents


def load_library(content_root: Path | None = None) -> Library:
    content_root = content_root or CONTENT
    topics = _documents(content_root, "topics")
    questions = _documents(content_root, "questions")

    errors: list[str] = []
    topic_map = _index(topics, "topic", errors)
    question_map = _index(questions, "question", errors)
    allowed_difficulties = {"beginner", "intermediate", "advanced"}

    for topic in topics:
        _required(topic, ["id", "title", "category", "difficulty", "quick_recall", "core_explanation"], errors)
        if topic.get("difficulty") not in allowed_difficulties:
            errors.append(f"{topic.get('_path')}: invalid difficulty '{topic.get('difficulty')}'")
        _normalize_topic(topic)
        for related in topic.get("related_topic_ids", []):
            if related not in topic_map:
                errors.append(f"{topic.get('_path')}: missing related topic '{related}'")
        for prerequisite in topic.get("prerequisite_topic_ids", []):
            if prerequisite not in topic_map:
                errors.append(f"{topic.get('_path')}: missing prerequisite topic '{prerequisite}'")
        _validate_sources(topic, errors)
        _validate_generation_metadata(topic, errors)
        _validate_mathematical_foundation(topic, errors)
        for image in topic.get("architecture", {}).get("images", []):
            image_path = ROOT / image.get("path", "")
            if not image.get("path") or not image_path.is_file():
                errors.append(f"{topic.get('_path')}: missing image '{image.get('path', '')}'")

    for question in questions:
        _normalize_question(question)
        _required(question, ["id", "topic_ids", "difficulty", "question", "direct_answer", "concept_refresher_topic_id"], errors)
        if question.get("difficulty") not in allowed_difficulties:
            errors.append(f"{question.get('_path')}: invalid difficulty '{question.get('difficulty')}'")
        references = question.get("topic_ids", []) + question.get("related_topic_ids", [])
        references.append(question.get("concept_refresher_topic_id"))
        for related in references:
            if related and related not in topic_map:
                errors.append(f"{question.get('_path')}: missing topic reference '{related}'")
        _validate_generation_metadata(question, errors)
    if errors:
        raise ContentError("\n".join(errors))
    return Library(topic_map, question_map)


def _index(items: list[dict[str, Any]], label: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for item in items:
        identifier = item.get("id")
        if not identifier:
            errors.append(f"{item.get('_path')}: missing {label} id")
        elif identifier in result:
            errors.append(f"duplicate {label} id '{identifier}'")
        else:
            result[identifier] = item
    return result


def _required(item: dict[str, Any], fields: list[str], errors: list[str]) -> None:
    for field in fields:
        if item.get(field) in (None, "", []):
            errors.append(f"{item.get('_path')}: missing required field '{field}'")


def _normalize_topic(topic: dict[str, Any]) -> None:
    """Keep V1 files readable while exposing V2's explicit relationship names."""
    topic.setdefault("content_version", 1)
    topic.setdefault("related_topic_ids", topic.get("related_topics", []))
    topic.setdefault("prerequisite_topic_ids", [])
    topic.setdefault("knowledge_type", ["conceptual"])


def _normalize_question(question: dict[str, Any]) -> None:
    """V2 calls this direct_answer; retain the V1 public field for old clients."""
    question.setdefault("content_version", 1)
    question.setdefault("direct_answer", question.get("short_answer", ""))
    question.setdefault("short_answer", question.get("direct_answer", ""))
    question.setdefault("question_type", "free_text")
    question.setdefault("question_category", "conceptual")
    question.setdefault("related_topic_ids", [])


def _validate_mathematical_foundation(topic: dict[str, Any], errors: list[str]) -> None:
    foundation = topic.get("mathematical_foundation")
    if foundation is None:
        return
    if not isinstance(foundation, dict):
        errors.append(f"{topic.get('_path')}: mathematical_foundation must be an object")
        return
    for section in foundation.get("sections", []):
        if not isinstance(section, dict) or not section.get("title") or not section.get("explanation"):
            errors.append(f"{topic.get('_path')}: mathematical foundation sections need title and explanation")
            continue
        for equation in section.get("equations", []):
            if not isinstance(equation, dict) or not equation.get("latex") or not equation.get("explanation"):
                errors.append(f"{topic.get('_path')}: equations need latex and explanation")


def _validate_sources(item: dict[str, Any], errors: list[str]) -> None:
    for source in item.get("sources", []):
        if not isinstance(source, dict) or not source.get("title") or not source.get("type"):
            errors.append(f"{item.get('_path')}: sources need title and type")


def _validate_generation_metadata(item: dict[str, Any], errors: list[str]) -> None:
    metadata = item.get("generation_metadata")
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        errors.append(f"{item.get('_path')}: generation_metadata must be an object")
    elif metadata.get("review_state") not in {"draft", "approved", "discarded"}:
        errors.append(f"{item.get('_path')}: generation_metadata has invalid review_state")
