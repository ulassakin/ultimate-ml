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
        for related in topic.get("related_topics", []):
            if related not in topic_map:
                errors.append(f"{topic.get('_path')}: missing related topic '{related}'")
        for image in topic.get("architecture", {}).get("images", []):
            image_path = ROOT / image.get("path", "")
            if not image.get("path") or not image_path.is_file():
                errors.append(f"{topic.get('_path')}: missing image '{image.get('path', '')}'")

    for question in questions:
        _required(question, ["id", "topic_ids", "difficulty", "question", "short_answer", "concept_refresher_topic_id"], errors)
        if question.get("difficulty") not in allowed_difficulties:
            errors.append(f"{question.get('_path')}: invalid difficulty '{question.get('difficulty')}'")
        references = question.get("topic_ids", []) + question.get("related_topic_ids", [])
        references.append(question.get("concept_refresher_topic_id"))
        for related in references:
            if related and related not in topic_map:
                errors.append(f"{question.get('_path')}: missing topic reference '{related}'")
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
