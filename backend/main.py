from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import hashlib
import json
import logging
import os
import re
import tempfile
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .content import ROOT, load_library
from .database import Database
from .ai.prompts import (QUESTION_INSTRUCTIONS, QUESTION_PROMPT_VERSION, TOPIC_INSTRUCTIONS, TOPIC_PROMPT_VERSION,
                         TOPIC_QUALITY_REVIEW_INSTRUCTIONS, TOPIC_QUALITY_REVIEW_PROMPT_VERSION,
                         YOUTUBE_CONCEPT_INSTRUCTIONS, YOUTUBE_CONCEPT_PROMPT_VERSION,
                         RELATIONSHIP_RESOLVER_INSTRUCTIONS, RELATIONSHIP_RESOLVER_PROMPT_VERSION)
from .ai.schemas import QuestionDraftBatch, QuestionDraftItem, RegeneratedSection, TopicDraft, TopicQualityReview, RelationshipResolution
from .ai.topic_quality import (apply_relationship_resolution, canonicalize_relationship_metadata, final_quality_report,
                               metadata_payload_consistency, normalize_topic, relationship_lint, taxonomy_context)
from .ai.metadata_resolution import MetadataResolutionService, compact_topic_payload
from .domain import Difficulty, ExplanationDepth
from .ai.service import AIService, AIUnavailableError, BudgetExceededError, StructuredOutputError
from .ai.structured import strict_response_schema
from .youtube.schemas import VideoConceptBatch
from .youtube.service import YoutubeService
from .youtube.transcript_provider import TranscriptUnavailableError, YouTubeTranscriptProvider

load_dotenv(ROOT / ".env")
app = FastAPI(title="Ultimate ML", version="0.1.0")
library = load_library()
database = Database(ROOT / "data" / "ultimate_ml.db")
ai_service = AIService(database)
youtube_service = YoutubeService(ROOT / "data" / "youtube_cache")
logger = logging.getLogger("ultimate_ml.local")


class ReviewInput(BaseModel):
    rating: Literal["again", "hard", "good", "easy"]
    answer: str = ""


class AISettingsInput(BaseModel):
    provider: Literal["openai"] | None = None
    model: str | None = None
    explanation_depth: ExplanationDepth | None = None
    enabled: bool | None = None
    monthly_budget_usd: float | None = None
    pricing_override: dict | None = None


class TopicDraftInput(BaseModel):
    title: str
    category: str = "classical_ml"
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    tags: list[str] = []
    focus: str = ""
    depth: ExplanationDepth = ExplanationDepth.ULTIMATE
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


class TopicEditInput(BaseModel):
    payload: dict
    edit_source: Literal["structured", "raw_json", "restore"] = "structured"


class QuestionEditInput(BaseModel):
    payload: dict
    edit_source: Literal["structured", "raw_json"] = "structured"


class RegenerateSectionInput(BaseModel):
    section: Literal["quick_recall", "big_picture", "why_it_exists", "intuition", "core_explanation", "mechanism", "ml_relevance", "practical_example", "mathematical_foundation", "common_misconceptions", "limitations", "mental_models", "deep_dive"]
    focus: str = ""


class ExistingDraftQualityReviewInput(BaseModel):
    # Normal runs are idempotent; force is an intentional, separately estimated re-review.
    force: bool = False


class YoutubeImportInput(BaseModel):
    video_url: str = ""
    pasted_transcript: str = ""
    title: str = ""
    channel: str = ""


class YoutubeExpansionInput(BaseModel):
    action: Literal["create", "enrich"]
    focus: str = ""


class YoutubeBatchSelection(BaseModel):
    concept_index: int
    action: Literal["create", "enrich"]


class YoutubeBatchInput(BaseModel):
    selections: list[YoutubeBatchSelection]


class YoutubeQuestionInput(BaseModel):
    topic_id: str
    focus: str = ""
    count: int = 5


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


def _topic_catalog():
    return [{"id": topic["id"], "title": topic["title"], "category": topic["category"],
             "summary": topic.get("one_sentence_summary", "")} for topic in library.topics.values()]


def _topic_quality_operations(operation_type: str) -> list[str]:
    return [operation_type, "youtube_topic_quality_review" if operation_type == "youtube_topic_expansion" else "topic_quality_review",
            "metadata_relationship_resolution"]


def _metadata_service() -> MetadataResolutionService:
    # Construct from the active database so isolated test databases remain isolated.
    return MetadataResolutionService(database)


def _resolve_topic_metadata(payload: dict, *, assigned_topic_id: str, metadata: dict | None = None) -> tuple[dict, dict, float, bool, list[dict], list[dict]]:
    """Local retrieval plus one resolver call; does not change educational fields."""
    service = _metadata_service()
    candidates = service.retrieve(payload, library.topics)
    cache_key, topic_hash, candidate_hashes = service.cache_key(payload, candidates, RELATIONSHIP_RESOLVER_PROMPT_VERSION)
    cached = database.get_metadata_resolution_cache(cache_key)
    result = None
    cost, reused = 0.0, bool(cached)
    if cached:
        result = cached["result"]
    elif not candidates:
        result = {"prerequisites": [], "related": [], "rejected_candidates": [], "resolver_version": RELATIONSHIP_RESOLVER_PROMPT_VERSION}
        database.set_metadata_resolution_cache(cache_key, topic_hash, candidate_hashes, RELATIONSHIP_RESOLVER_PROMPT_VERSION, result)
        reused = True
    else:
        request = {"current_topic": compact_topic_payload(payload), "retrieved_candidates": candidates,
                   "taxonomy": taxonomy_context(), "relationship_definitions": {
                       "prerequisite": "Material learning dependency; lack of it substantially blocks or confuses understanding.",
                       "related": "Strong neighbor, extension, alternative, comparison, or central architecture/objective pairing."}}
        try:
            resolved, _, cost = ai_service.generate(operation_type="metadata_relationship_resolution",
                instructions=RELATIONSHIP_RESOLVER_INSTRUCTIONS, input_text=json.dumps(request),
                schema_name="ultimate_ml_relationship_resolution", schema=strict_response_schema(RelationshipResolution),
                validate=RelationshipResolution.model_validate, max_output_tokens=1600,
                metadata={"prompt_version": RELATIONSHIP_RESOLVER_PROMPT_VERSION, "topic_content_hash": topic_hash,
                          "retrieved_candidate_count": len(candidates), **(metadata or {})})
        except Exception as exc:
            _ai_error(exc)
        result = {**resolved.model_dump(), "resolver_version": RELATIONSHIP_RESOLVER_PROMPT_VERSION}
        database.set_metadata_resolution_cache(cache_key, topic_hash, candidate_hashes, RELATIONSHIP_RESOLVER_PROMPT_VERSION, result)
    resolved_payload, warnings, blocking = apply_relationship_resolution(payload, result, candidates=candidates, assigned_topic_id=assigned_topic_id)
    resolved_payload["metadata_resolution"].update({"cache_key": cache_key, "cached": reused,
        "topic_hash": topic_hash, "candidate_hashes": candidate_hashes, "resolver_prompt_version": RELATIONSHIP_RESOLVER_PROMPT_VERSION})
    return resolved_payload, result, cost, reused, warnings, blocking


def _queue_status_for_draft(state: str) -> str:
    return {"draft": "ready", "approved": "approved", "discarded": "discarded"}.get(state, "failed")


def _sync_video_drafts_to_queue(import_id: str, analysis: dict | None = None) -> None:
    """Add queue projections for old/new drafts without changing the draft itself."""
    concept_lookup = {_slug(item.get("canonical_name", "")): item for item in (analysis or {}).get("concepts", [])}
    maximum = ai_service.estimate_operations(_topic_quality_operations("youtube_topic_expansion"))["maximum_estimated_cost_usd"]
    for draft in database.list_video_drafts(import_id):
        concept_name = draft["metadata"].get("youtube_concept")
        if not concept_name:
            continue
        key = _slug(concept_name)
        concept = concept_lookup.get(key) or {"canonical_name": concept_name,
            "source_evidence_summary": draft["payload"].get("source_provenance", {}).get("source_derived", {}).get("source_evidence_summary", "Existing video-derived draft."),
            "ml_learning_value": "Existing video-derived draft.", "timestamp_seconds": []}
        item = database.create_queue_item(str(uuid4()), youtube_import_id=import_id, canonical_concept=key,
            concept=concept, action="enrich" if draft["metadata"].get("enrich_existing_topic_id") else "create",
            status=_queue_status_for_draft(draft["state"]), draft_id=draft["id"], maximum_cost_usd=maximum)
        database.update_queue_item(item["id"], status=_queue_status_for_draft(draft["state"]), draft_id=draft["id"])


def _decorate_analysis(import_id: str, analysis: dict | None) -> dict | None:
    if not analysis:
        return analysis
    decorated = json.loads(json.dumps(analysis))
    _sync_video_drafts_to_queue(import_id, decorated)
    for concept in decorated.get("concepts", []):
        active = database.find_active_video_draft(import_id, concept["canonical_name"])
        queue_item = database.get_queue_item_by_concept(import_id, _slug(concept["canonical_name"]))
        if active:
            concept["draft"] = {"id": active["id"], "state": active["state"], "title": active["title"]}
            concept["allowed_actions"] = ["review"]
        elif queue_item and queue_item["status"] in {"pending", "generating", "failed"}:
            concept["queue"] = {"id": queue_item["id"], "status": queue_item["status"], "error_message": queue_item["error_message"]}
            concept["allowed_actions"] = ["retry"] if queue_item["status"] == "failed" else []
    return decorated


def _public_youtube_import(record: dict) -> dict:
    """Expose metadata and analysis, never the cached full third-party transcript."""
    document = youtube_service.load_transcript(record["transcript_cache_path"])
    return {key: value for key, value in record.items() if key not in {"transcript_cache_path", "analysis"}} | {
        "analysis": _decorate_analysis(record["id"], record.get("analysis")),
        "transcript": {"segment_count": len(document.segments), "preview": youtube_service.preview(document),
                       "cached_locally": True, "public_content_saved": False},
        "usage": database.youtube_usage(record["id"]),
    }


def _source_context(document, concept: dict) -> dict:
    return {"kind": document.source.kind, "video_url": document.source.video_url,
            "video_id": document.source.video_id, "title": document.source.title,
            "channel": document.source.channel, "attribution": document.source.transcript_attribution,
            "concept": concept["canonical_name"], "source_evidence_summary": concept["source_evidence_summary"],
            "timestamp_seconds": concept.get("timestamp_seconds", [])}


def _video_concept(record: dict, concept_index: int) -> dict:
    if not record.get("analysis"):
        raise HTTPException(404, "Analyze a video import before generating drafts")
    concepts = record["analysis"].get("concepts", [])
    if concept_index < 0 or concept_index >= len(concepts):
        raise HTTPException(404, "Video concept not found")
    return concepts[concept_index]


def _generate_video_concept_draft(record: dict, concept: dict, action: str, focus: str = "") -> dict:
    existing_draft = database.find_active_video_draft(record["id"], concept["canonical_name"])
    if existing_draft:
        return {"id": existing_draft["id"], "state": existing_draft["state"], "payload": existing_draft["payload"], "reused": True,
                "usage": {"estimated_cost_usd": 0}, "budget": ai_service.usage_summary()}
    match = concept.get("existing_topic_match")
    if action == "create" and match:
        raise HTTPException(409, "This concept already matches an existing topic. Choose Enrich existing or Ignore.")
    if action == "enrich" and not match:
        raise HTTPException(422, "Only a deterministically matched existing topic can be enriched.")
    document = youtube_service.load_transcript(record["transcript_cache_path"])
    if match:
        existing = library.topics.get(match["id"])
        if not existing:
            raise HTTPException(422, "The matched topic is no longer in the library. Reanalyze the import.")
        topic_input = TopicDraftInput(title=existing["title"], category=existing["category"], difficulty=existing["difficulty"],
            tags=existing.get("tags", []), focus=focus or f"Enrich this topic using the video’s treatment of {concept['canonical_name']}.", allow_duplicate=True)
        return _generate_topic_draft(topic_input, operation_type="youtube_topic_expansion", source_context=_source_context(document, concept) | {"base_topic": public(existing)},
            extra_metadata={"youtube_import_id": record["id"], "youtube_concept": concept["canonical_name"], "enrich_existing_topic_id": existing["id"]}, duplicate_check=False)
    topic_input = TopicDraftInput(title=concept["canonical_name"], category="ml_fundamentals", difficulty=Difficulty.INTERMEDIATE,
        focus=focus or f"Build a deeper ML learning topic from this source concept: {concept['ml_learning_value']}.")
    return _generate_topic_draft(topic_input, operation_type="youtube_topic_expansion", source_context=_source_context(document, concept),
        extra_metadata={"youtube_import_id": record["id"], "youtube_concept": concept["canonical_name"]})


def _generate_topic_draft(body: TopicDraftInput, *, operation_type="topic_draft", source_context=None,
                          extra_metadata=None, duplicate_check=True):
    duplicate = _title_duplicate(body.title)
    if duplicate and duplicate_check and not body.allow_duplicate:
        raise HTTPException(409, {"message": "A matching topic already exists. Open it or explicitly create a separate draft.", "duplicate": duplicate})
    assigned_topic_id = _slug(body.title)
    catalog = _topic_catalog()
    operation_types = _topic_quality_operations(operation_type)
    try:
        preflight = ai_service.require_budget_for_operations(operation_types)
    except Exception as exc:
        _ai_error(exc)
    prompt = {"title": body.title, "primary_category": body.category, "difficulty": body.difficulty.value,
              "tags": body.tags, "focus": body.focus, "explanation_depth": body.depth.value,
              "include_mathematics": body.include_mathematics, "include_practical_examples": body.include_examples,
              "include_misconceptions": body.include_misconceptions, "suggest_related_topics": body.suggest_related_topics,
              "concept_type_options": ["broad_concept", "named_method", "architecture", "loss_or_objective", "mathematical_concept", "training_mechanism", "evaluation_concept"],
              "taxonomy": taxonomy_context(),
              "relationship_rules": "Do not select catalog IDs. Leave durable relationship arrays empty; a separate metadata resolver handles sparse graph edges after quality review.",
              "future_source_context": source_context or "None supplied. Do not invent sources."}
    try:
        draft, result, cost = ai_service.generate(operation_type=operation_type, instructions=TOPIC_INSTRUCTIONS,
            input_text=json.dumps(prompt), schema_name="ultimate_ml_topic_draft", schema=strict_response_schema(TopicDraft),
            validate=TopicDraft.model_validate, max_output_tokens=5000,
            metadata={"prompt_version": TOPIC_PROMPT_VERSION, **(extra_metadata or {})})
    except Exception as exc:
        _ai_error(exc)
    payload, initial_warnings, initial_blocking = normalize_topic(draft.model_dump(), requested_title=body.title,
        assigned_topic_id=assigned_topic_id, catalog=catalog, source_context=source_context)
    review_request = {"requested_title": body.title, "requested_category": body.category, "requested_difficulty": body.difficulty.value,
        "focus": body.focus, "taxonomy": taxonomy_context(), "candidate": payload,
        "source_context": source_context or "None supplied"}
    review_operation = operation_types[1]
    try:
        reviewed, review_result, review_cost = ai_service.generate(operation_type=review_operation,
            instructions=TOPIC_QUALITY_REVIEW_INSTRUCTIONS, input_text=json.dumps(review_request),
            schema_name="ultimate_ml_topic_quality_review", schema=strict_response_schema(TopicQualityReview),
            validate=TopicQualityReview.model_validate, max_output_tokens=5000,
            metadata={"prompt_version": TOPIC_QUALITY_REVIEW_PROMPT_VERSION, "generated_draft_title": body.title, **(extra_metadata or {})})
    except Exception as exc:
        _ai_error(exc)
    payload, review_warnings, review_blocking = normalize_topic(reviewed.corrected_topic.model_dump(), requested_title=body.title,
        assigned_topic_id=assigned_topic_id, catalog=catalog, source_context=source_context)
    if source_context:
        base_topic = source_context.get("base_topic")
        if base_topic:
            # Enrichment must not erase reviewed architecture/pipeline assets.
            for field in ("architecture", "why_it_matters", "knowledge_type", "related_topics"):
                if field in base_topic:
                    payload[field] = base_topic[field]
    payload, resolution, resolver_cost, resolver_reused, resolver_warnings, resolver_blocking = _resolve_topic_metadata(
        payload, assigned_topic_id=assigned_topic_id, metadata={"generated_draft_title": body.title, **(extra_metadata or {})})
    quality_report = final_quality_report(reviewed.quality_report.model_dump(), initial_warnings + review_warnings + resolver_warnings,
        review_blocking + resolver_blocking, payload=payload)
    consistency_blocking = metadata_payload_consistency(payload, quality_report, catalog=catalog, assigned_topic_id=assigned_topic_id)
    quality_report["blocking_issues_remaining"] += consistency_blocking
    # Initial lint findings were deliberately presented to the repairer. They are
    # history of what the gate caught, not blockers once the corrected topic passes.
    quality_report["blocking_issues_fixed"] = initial_blocking + quality_report["blocking_issues_fixed"]
    quality_status = "needs_attention" if quality_report["blocking_issues_remaining"] else "ready"
    payload["quality_report"] = quality_report
    payload["quality_status"] = quality_status
    payload["generation_metadata"] = {"generated_by_ai": True, "provider": "openai", "model": ai_service.settings().model,
        "prompt_version": TOPIC_PROMPT_VERSION, "quality_review_prompt_version": TOPIC_QUALITY_REVIEW_PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_focus": body.focus, "review_state": "draft", **(extra_metadata or {})}
    draft_id = str(uuid4())
    total_cost = cost + review_cost + resolver_cost
    database.create_draft(draft_id, "topic", payload["title"], payload, {"request_cost_usd": total_cost, "quality_review_cost_usd": review_cost,
        "metadata_resolution_cost_usd": resolver_cost, **(extra_metadata or {})})
    return {"id": draft_id, "state": "draft", "payload": payload, "usage": {"input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens, "estimated_cost_usd": total_cost,
            "generation_estimated_cost_usd": cost, "quality_review_estimated_cost_usd": review_cost,
            "metadata_resolution_estimated_cost_usd": resolver_cost, "metadata_resolution_cached": resolver_reused,
            "maximum_estimated_cost_usd": preflight["maximum_estimated_cost_usd"]}, "budget": ai_service.usage_summary()}


def _sanitize_draft_relationships(payload: dict) -> dict:
    """Never let an AI-generated relationship point at an unknown topic."""
    # Manual edits from prior releases may not have relationship rationales; only
    # generated drafts use the stricter normalization gate above.
    allowed = set(library.topics)
    copy = dict(payload)
    warnings = list(copy.get("relationship_warnings", []))
    for field in ("prerequisite_topic_ids", "related_topic_ids"):
        requested = copy.get(field, [])
        invalid = [item for item in requested if item not in allowed]
        copy[field] = list(dict.fromkeys(item for item in requested if item in allowed))
        if invalid:
            warnings.append({"field": field, "message": f"Removed unavailable topic IDs: {', '.join(invalid)}"})
    copy["relationship_warnings"] = warnings
    return canonicalize_relationship_metadata(copy)


def _payload_hash(payload: dict) -> str:
    """Hash authoring content, not transient review UI metadata."""
    copy = json.loads(json.dumps(payload))
    for field in ("existing_quality_review", "quality_report", "quality_status", "relationship_warnings", "durable_topic_id"):
        copy.pop(field, None)
    encoded = json.dumps(copy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _draft_quality_state(payload: dict) -> dict:
    existing = payload.get("existing_quality_review")
    if isinstance(existing, dict):
        return existing
    # Pre-gate drafts remain untouched; this response-only default makes their
    # upgrade path visible without writing to any already-paid draft.
    if payload.get("quality_status") == "needs_attention":
        return {"status": "needs_attention", "reviewer_prompt_version": payload.get("generation_metadata", {}).get("quality_review_prompt_version")}
    if payload.get("quality_report"):
        return {"status": "reviewed", "reviewer_prompt_version": payload.get("generation_metadata", {}).get("quality_review_prompt_version")}
    return {"status": "not_reviewed", "reviewer_prompt_version": None}


def _draft_public_quality(draft: dict) -> dict:
    return {**draft, "quality_review": _draft_quality_state(draft["payload"])}


def _existing_review_source_context(payload: dict) -> dict | None:
    source = payload.get("source_provenance", {}).get("source_derived")
    if not isinstance(source, dict) or not source.get("video_url") or not source.get("title"):
        return None
    return source


def _run_existing_draft_quality_review(draft: dict, *, force: bool = False) -> dict:
    if draft["draft_type"] != "topic" or draft["state"] != "draft":
        raise HTTPException(409, "Only active topic drafts can be quality-reviewed.")
    original = json.loads(json.dumps(draft["payload"]))
    source_hash = _payload_hash(original)
    current_state = _draft_quality_state(original)
    reviewer_version = TOPIC_QUALITY_REVIEW_PROMPT_VERSION
    if not force and current_state.get("status") in {"reviewed", "needs_attention"} and current_state.get("reviewer_prompt_version") == reviewer_version and current_state.get("reviewed_payload_hash") == source_hash:
        return {"draft": _draft_public_quality(draft), "reused": True, "reason": "This exact repaired revision already passed the current reviewer.", "budget": ai_service.usage_summary()}
    previous = database.find_completed_draft_quality_review(draft["id"], source_hash, reviewer_version)
    if not force and previous:
        return {"draft": _draft_public_quality(draft), "reused": True, "reason": "This exact source revision already has a saved quality review. Restore its repaired revision instead of paying again.", "existing_revision_id": previous["id"], "budget": ai_service.usage_summary()}
    try:
        preflight = ai_service.require_budget_for_operations(["topic_quality_review_existing", "metadata_relationship_resolution"])
    except Exception as exc:
        _ai_error(exc)
    database.create_draft_quality_revision(str(uuid4()), draft["id"], "pre_quality_review", original,
        source_payload_hash=source_hash, reviewer_prompt_version=reviewer_version)
    reviewing = {**original, "existing_quality_review": {"status": "reviewing", "source_payload_hash": source_hash,
        "reviewer_prompt_version": reviewer_version, "started_at": datetime.now(timezone.utc).isoformat()}}
    database.update_draft(draft["id"], reviewing)
    request = {"mode": "existing_paid_draft_quality_review", "instruction": "Review this existing full draft directly. Do not generate a new topic, questions, or IDs.",
        "taxonomy": taxonomy_context(), "candidate": original,
        "source_context": _existing_review_source_context(original) or "No source context supplied."}
    try:
        reviewed, result, cost = ai_service.generate(operation_type="topic_quality_review_existing",
            instructions=TOPIC_QUALITY_REVIEW_INSTRUCTIONS, input_text=json.dumps(request),
            schema_name="ultimate_ml_topic_quality_review", schema=strict_response_schema(TopicQualityReview),
            validate=TopicQualityReview.model_validate, max_output_tokens=5000,
            metadata={"prompt_version": reviewer_version, "draft_id": draft["id"], "source_payload_hash": source_hash,
                      "existing_paid_draft": True, "forced_re_review": force})
    except Exception as exc:
        failed = {**original, "existing_quality_review": {"status": "review_failed", "source_payload_hash": source_hash,
            "reviewer_prompt_version": reviewer_version, "failed_at": datetime.now(timezone.utc).isoformat(),
            "message": "Quality review failed. The original draft is preserved; retry when Settings are available."}}
        database.update_draft(draft["id"], failed)
        _ai_error(exc)
    corrected = reviewed.corrected_topic.model_dump()
    # The review schema intentionally owns topic fields only. Preserve durable
    # provenance/assets and all other non-authoring fields from the paid draft.
    authored_fields = set(TopicDraft.model_fields)
    extras = {key: value for key, value in original.items() if key not in authored_fields and key not in {
        "existing_quality_review", "quality_report", "quality_status", "relationship_warnings", "durable_topic_id"}}
    corrected = {**corrected, **extras}
    corrected, warnings, blocking = normalize_topic(corrected, requested_title=original.get("title", draft["title"]),
        assigned_topic_id=_slug(original.get("title", draft["title"])), catalog=_topic_catalog(),
        source_context=_existing_review_source_context(original))
    corrected, resolution, resolver_cost, resolver_reused, resolver_warnings, resolver_blocking = _resolve_topic_metadata(
        corrected, assigned_topic_id=_slug(original.get("title", draft["title"])), metadata={"draft_id": draft["id"], "existing_paid_draft": True})
    report = final_quality_report(reviewed.quality_report.model_dump(), warnings + resolver_warnings, blocking + resolver_blocking, payload=corrected)
    report["blocking_issues_remaining"] += metadata_payload_consistency(corrected, report, catalog=_topic_catalog(),
        assigned_topic_id=_slug(original.get("title", draft["title"])))
    status = "needs_attention" if report["blocking_issues_remaining"] else "reviewed"
    corrected["quality_report"] = report
    corrected["quality_status"] = "needs_attention" if status == "needs_attention" else "ready"
    corrected_hash = _payload_hash(corrected)
    corrected["existing_quality_review"] = {"status": status, "source_payload_hash": source_hash,
        "reviewed_payload_hash": corrected_hash, "reviewer_prompt_version": reviewer_version,
        "reviewed_at": datetime.now(timezone.utc).isoformat(), "forced_re_review": force}
    revision = database.create_draft_quality_revision(str(uuid4()), draft["id"], "quality_review", corrected,
        source_payload_hash=source_hash, reviewer_prompt_version=reviewer_version, quality_report=report)
    database.update_draft(draft["id"], corrected)
    saved = database.get_draft(draft["id"])
    return {"draft": _draft_public_quality(saved), "reused": False, "revision": revision,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "estimated_cost_usd": cost + resolver_cost,
                  "quality_review_estimated_cost_usd": cost, "metadata_resolution_estimated_cost_usd": resolver_cost,
                  "metadata_resolution_cached": resolver_reused,
                  "maximum_estimated_cost_usd": preflight["maximum_estimated_cost_usd"]}, "budget": ai_service.usage_summary()}


def _validate_topic_draft_payload(payload: dict) -> dict:
    errors = []
    try:
        draft = TopicDraft.model_validate(payload)
    except Exception as exc:
        return {"valid": False, "errors": [{"field": "draft", "message": str(exc)}], "warnings": payload.get("relationship_warnings", [])}
    for field in ("prerequisite_topic_ids", "related_topic_ids"):
        invalid = [item for item in getattr(draft, field) if item not in library.topics]
        if invalid:
            errors.append({"field": field, "message": f"Unknown topic IDs: {', '.join(invalid)}"})
    relationship_errors, relationship_warnings = relationship_lint(payload, catalog=_topic_catalog(), assigned_topic_id=_slug(draft.title))
    errors.extend({"field": "relationships", "message": item["message"]} for item in relationship_errors)
    if draft.category not in taxonomy_context():
        errors.append({"field": "category", "message": "Choose a category from the local taxonomy."})
    if payload.get("quality_status") == "ready" and payload.get("metadata_resolution"):
        consistency = metadata_payload_consistency(payload, payload.get("quality_report", {}), catalog=_topic_catalog(),
            assigned_topic_id=_slug(draft.title))
        errors.extend({"field": "metadata", "message": item["message"]} for item in consistency)
    return {"valid": not errors, "errors": errors, "warnings": payload.get("relationship_warnings", []) + relationship_warnings}


def _reload_library():
    global library
    library = load_library()


def _approved_topic(payload: dict, *, canonical_id: str | None = None) -> dict:
    topic = dict(payload)
    topic.pop("relationship_warnings", None)
    topic.pop("durable_topic_id", None)
    # The request/model payload never gets to select a durable ID. A stable ID is
    # passed only for a deliberate enrichment target.
    topic_id = canonical_id or _slug(topic["title"])
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


def _atomic_write_json(path: Path, data: dict):
    """A durable local edit either fully replaces a JSON file or leaves it untouched."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _revision_directory(kind: str) -> Path:
    path = ROOT / "data" / f"{kind}_revisions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_revision(kind: str, identifier: str, payload: dict, edit_source: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = _revision_directory(kind) / f"{_slug(identifier)}-{timestamp}-{edit_source}.json"
    _atomic_write_json(path, payload)
    return path


def _validate_topic_edit(topic_id: str, payload: dict) -> dict:
    errors = []
    for field in ("title", "category", "difficulty", "quick_recall", "core_explanation"):
        if payload.get(field) in (None, "", []):
            errors.append({"field": field, "message": "This field is required."})
    if payload.get("difficulty") not in {item.value for item in Difficulty}:
        errors.append({"field": "difficulty", "message": "Must be beginner, intermediate, or advanced."})
    for field in ("prerequisite_topic_ids", "related_topic_ids"):
        unknown = [value for value in payload.get(field, []) if value not in library.topics]
        if unknown:
            errors.append({"field": field, "message": f"Unknown topic IDs: {', '.join(unknown)}"})
    for index, source in enumerate(payload.get("sources", [])):
        if not isinstance(source, dict) or not source.get("title") or not source.get("type"):
            errors.append({"field": f"sources[{index}]", "message": "Sources need a title and type."})
    provenance = payload.get("source_provenance")
    if provenance is not None and (not isinstance(provenance, dict) or not isinstance(provenance.get("source_derived"), dict)
                                   or not isinstance(provenance.get("ai_expanded"), str)):
        errors.append({"field": "source_provenance", "message": "Source provenance needs source_derived and ai_expanded fields."})
    try:
        if payload.get("content_version", 2) >= 2:
            TopicDraft.model_validate(payload)
    except Exception as exc:
        errors.append({"field": "schema", "message": str(exc)})
    return {"valid": not errors, "errors": errors}


def _save_edited_topic(topic_id: str, payload: dict, edit_source: str) -> dict:
    # An editor can omit or accidentally alter this hidden implementation detail;
    # changing a durable ID is a separate migration, never an ordinary content edit.
    payload = {**payload, "id": topic_id}
    validation = _validate_topic_edit(topic_id, payload)
    if not validation["valid"]:
        raise HTTPException(422, {"message": "Fix topic validation errors before saving.", **validation})
    existing = library.topics.get(topic_id)
    if not existing:
        raise HTTPException(404, "Topic not found")
    target = ROOT / existing["_path"]
    previous = json.loads(target.read_text(encoding="utf-8"))
    _save_revision("topic", topic_id, previous, edit_source)
    try:
        _atomic_write_json(target, payload)
        _reload_library()
    except Exception as exc:
        _atomic_write_json(target, previous)
        _reload_library()
        raise HTTPException(422, f"Topic was not saved: {exc}") from exc
    return public(library.topics[topic_id])


def _question_matches_topic(question: dict, topic_id: str) -> bool:
    return topic_id in question.get("topic_ids", []) or question.get("concept_refresher_topic_id") == topic_id


def _question_candidate_status(payload: dict) -> dict:
    topic_id = payload.get("topic_id")
    existing = {(_slug(question["question"]), question["id"]): question for question in library.questions.values()}
    normalized = json.loads(json.dumps(payload))
    for candidate in normalized.get("questions", []):
        key = (_slug(candidate.get("question", "")), topic_id)
        matching = next((question for (text, _), question in existing.items()
                         if text == key[0] and _question_matches_topic(question, topic_id)), None)
        if matching:
            candidate["approval_status"] = "approved"
            candidate["approved_question_id"] = matching["id"]
        else:
            candidate.setdefault("approval_status", "selected" if candidate.get("selected") else "draft")
    statuses = {item.get("approval_status") for item in normalized.get("questions", [])}
    normalized["approval_state"] = "approved" if statuses == {"approved"} and statuses else ("draft" if statuses <= {"draft", "selected"} else "partial")
    return normalized


def _validate_question_candidate(candidate: dict, topic_id: str) -> tuple[dict | None, list[str]]:
    try:
        item = QuestionDraftItem.model_validate(candidate)
    except Exception as exc:
        return None, [str(exc)]
    output = item.model_dump()
    unknown = [value for value in output.get("related_topic_ids", []) if value not in library.topics]
    if unknown:
        output["relationship_warnings"] = [f"Removed unavailable related topic IDs: {', '.join(unknown)}"]
        output["related_topic_ids"] = [value for value in output["related_topic_ids"] if value in library.topics]
    return output, []


def _next_question_id(topic_id: str, reserved: set[str]) -> str:
    numbers = []
    pattern = re.compile(rf"^{re.escape(topic_id)}_q(\d+)$")
    for question_id in set(library.questions) | reserved:
        matched = pattern.match(question_id)
        if matched:
            numbers.append(int(matched.group(1)))
    return f"{topic_id}_q{max(numbers, default=0) + 1:03d}"


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


@app.get("/api/topics/{topic_id}/editable")
def editable_topic(topic_id: str):
    if topic_id not in library.topics:
        raise HTTPException(404, "Topic not found")
    path = ROOT / library.topics[topic_id]["_path"]
    return {"topic": public(library.topics[topic_id]), "raw_json": path.read_text(encoding="utf-8")}


@app.put("/api/topics/{topic_id}")
def edit_topic(topic_id: str, body: TopicEditInput):
    return _save_edited_topic(topic_id, body.payload, body.edit_source)


def _metadata_rebuild_estimate(payload: dict) -> dict:
    service = _metadata_service()
    candidates = service.retrieve(payload, library.topics)
    cache_key, _, _ = service.cache_key(payload, candidates, RELATIONSHIP_RESOLVER_PROMPT_VERSION)
    cached = bool(database.get_metadata_resolution_cache(cache_key)) or not candidates
    estimate = ai_service.estimate_operations(["metadata_relationship_resolution"])
    return {**estimate, "candidate_count": len(candidates), "resolver_prompt_version": RELATIONSHIP_RESOLVER_PROMPT_VERSION,
            "cached": cached, "maximum_estimated_cost_usd": 0 if cached else estimate["maximum_estimated_cost_usd"]}


@app.get("/api/topics/{topic_id}/metadata-rebuild-estimate")
def topic_metadata_rebuild_estimate(topic_id: str):
    if topic_id not in library.topics:
        raise HTTPException(404, "Topic not found")
    return _metadata_rebuild_estimate(public(library.topics[topic_id]))


@app.post("/api/topics/{topic_id}/rebuild-relationships")
def rebuild_topic_relationships(topic_id: str):
    if topic_id not in library.topics:
        raise HTTPException(404, "Topic not found")
    existing = library.topics[topic_id]
    payload = public(existing)
    estimate = _metadata_rebuild_estimate(payload)
    if not estimate["cached"]:
        try:
            ai_service.require_budget_for_operations(["metadata_relationship_resolution"])
        except Exception as exc:
            _ai_error(exc)
    rebuilt, _, cost, cached, warnings, blocking = _resolve_topic_metadata(payload, assigned_topic_id=topic_id,
        metadata={"topic_id": topic_id, "metadata_only_rebuild": True})
    report = rebuilt.get("quality_report")
    if isinstance(report, dict):
        report = final_quality_report(report, warnings, blocking, payload=rebuilt)
        report["blocking_issues_remaining"] += metadata_payload_consistency(rebuilt, report, catalog=_topic_catalog(), assigned_topic_id=topic_id)
        rebuilt["quality_report"] = report
        rebuilt["quality_status"] = "needs_attention" if report["blocking_issues_remaining"] else "ready"
    # This is an explicit metadata-only revision; educational prose, provenance,
    # questions, and the durable ID are copied unchanged.
    result = _save_edited_topic(topic_id, rebuilt, "metadata-rebuild")
    return {"topic": result, "usage": {"estimated_cost_usd": cost, "maximum_estimated_cost_usd": estimate["maximum_estimated_cost_usd"],
            "cached": cached}, "warnings": warnings, "budget": ai_service.usage_summary()}


@app.get("/api/topics/{topic_id}/revisions")
def topic_revisions(topic_id: str):
    if topic_id not in library.topics:
        raise HTTPException(404, "Topic not found")
    revisions = sorted(_revision_directory("topic").glob(f"{_slug(topic_id)}-*.json"), reverse=True)
    return [{"name": path.name, "created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()} for path in revisions]


@app.post("/api/topics/{topic_id}/revisions/{revision_name}/restore")
def restore_topic_revision(topic_id: str, revision_name: str):
    if "/" in revision_name or "\\" in revision_name:
        raise HTTPException(422, "Invalid revision name")
    path = _revision_directory("topic") / revision_name
    if not path.is_file() or not path.name.startswith(f"{_slug(topic_id)}-"):
        raise HTTPException(404, "Topic revision not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "Stored revision is invalid JSON") from exc
    return _save_edited_topic(topic_id, payload, "restore")


@app.get("/api/topics/{topic_id}/question-management")
def topic_question_management(topic_id: str):
    if topic_id not in library.topics:
        raise HTTPException(404, "Topic not found")
    approved = [public(question) for question in library.questions.values() if _question_matches_topic(question, topic_id) and not question.get("archived")]
    pending = []
    for draft in database.list_drafts_by_type("question"):
        payload = _question_candidate_status(draft["payload"])
        if payload.get("topic_id") == topic_id and draft["state"] not in {"discarded", "approved"} and payload["approval_state"] != "approved":
            pending.append({"id": draft["id"], "state": draft["state"], "approval_state": payload["approval_state"],
                            "candidate_count": len(payload.get("questions", [])), "approved_candidate_count": sum(item.get("approval_status") == "approved" for item in payload.get("questions", []))})
    return {"topic_id": topic_id, "approved_count": len(approved), "approved_questions": approved, "pending_drafts": pending}


@app.put("/api/questions/{question_id}")
def edit_approved_question(question_id: str, body: QuestionEditInput):
    existing = library.questions.get(question_id)
    if not existing:
        raise HTTPException(404, "Question not found")
    payload = body.payload
    errors = []
    if payload.get("id") != question_id:
        errors.append({"field": "id", "message": "Question ID is stable during local edits."})
    if payload.get("topic_ids") != existing.get("topic_ids") or payload.get("concept_refresher_topic_id") != existing.get("concept_refresher_topic_id"):
        errors.append({"field": "references", "message": "Topic and refresher references are preserved during question edits."})
    candidate, candidate_errors = _validate_question_candidate(payload, existing["concept_refresher_topic_id"])
    errors.extend({"field": "schema", "message": error} for error in candidate_errors)
    if errors:
        raise HTTPException(422, {"message": "Fix question validation errors before saving.", "errors": errors})
    candidate.update({"id": question_id, "topic_ids": existing["topic_ids"], "concept_refresher_topic_id": existing["concept_refresher_topic_id"],
                      "content_version": existing.get("content_version", 2), "generation_metadata": existing.get("generation_metadata", {})})
    target = ROOT / existing["_path"]
    previous = json.loads(target.read_text(encoding="utf-8"))
    _save_revision("question", question_id, previous, body.edit_source)
    try:
        _atomic_write_json(target, candidate)
        _reload_library()
    except Exception as exc:
        _atomic_write_json(target, previous); _reload_library()
        raise HTTPException(422, f"Question was not saved: {exc}") from exc
    return public(library.questions[question_id])


@app.post("/api/questions/{question_id}/archive")
def archive_question(question_id: str):
    existing = library.questions.get(question_id)
    if not existing:
        raise HTTPException(404, "Question not found")
    target = ROOT / existing["_path"]
    previous = json.loads(target.read_text(encoding="utf-8"))
    _save_revision("question", question_id, previous, "archive")
    updated = {**previous, "archived": True}
    _atomic_write_json(target, updated); _reload_library()
    return {"id": question_id, "archived": True}


@app.get("/api/questions")
def questions():
    return [public(question) for question in library.questions.values() if not question.get("archived")]


@app.get("/api/review/due")
def due_questions():
    progress, _ = database.progress()
    states = {row["question_id"]: row for row in progress}
    now = datetime.now(timezone.utc).isoformat()
    due = [question for question in library.questions.values() if not question.get("archived")
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


@app.post("/api/youtube/imports")
def create_youtube_import(body: YoutubeImportInput):
    if bool(body.video_url.strip()) == bool(body.pasted_transcript.strip()):
        raise HTTPException(422, "Provide exactly one source: a YouTube URL or a pasted transcript.")
    try:
        if body.pasted_transcript.strip():
            document = youtube_service.pasted_transcript(body.pasted_transcript, title=body.title,
                video_url=body.video_url, channel=body.channel)
        else:
            document = YouTubeTranscriptProvider().retrieve(body.video_url, title=body.title, channel=body.channel)
    except (ValueError, TranscriptUnavailableError) as exc:
        raise HTTPException(422, str(exc)) from exc
    import_id = str(uuid4())
    path = youtube_service.save_transcript(import_id, document)
    database.create_youtube_import(import_id, document.source.model_dump(), str(path))
    return _public_youtube_import(database.get_youtube_import(import_id))


@app.get("/api/youtube/imports")
def list_youtube_imports():
    imports = []
    for record in database.list_youtube_imports():
        _sync_video_drafts_to_queue(record["id"], record.get("analysis"))
        queue = database.list_queue_items(record["id"])
        imports.append({"id": record["id"], "state": record["state"], "source": record["source"],
            "created_at": record["created_at"], "updated_at": record["updated_at"],
            "concept_count": len((record.get("analysis") or {}).get("concepts", [])),
            "draft_count": len([item for item in queue if item.get("draft_id")]), "queue_count": len(queue)})
    return imports


@app.get("/api/youtube/imports/{import_id}")
def get_youtube_import(import_id: str):
    record = database.get_youtube_import(import_id)
    if not record:
        raise HTTPException(404, "Video import not found")
    return _public_youtube_import(record)


@app.get("/api/youtube/imports/{import_id}/estimate")
def youtube_analysis_estimate(import_id: str):
    record = database.get_youtube_import(import_id)
    if not record:
        raise HTTPException(404, "Video import not found")
    chunks = youtube_service.chunks(youtube_service.load_transcript(record["transcript_cache_path"]))
    per_call = ai_service.estimate_maximum("youtube_concept_extraction")
    return {"operation_type": "youtube_concept_extraction", "chunk_count": len(chunks),
            "maximum_estimated_cost_usd": per_call * len(chunks), "remaining_budget_usd": ai_service.usage_summary()["remaining_budget_usd"]}


@app.get("/api/ai/topic-draft-estimate")
def topic_draft_estimate(youtube: bool = False):
    operations = _topic_quality_operations("youtube_topic_expansion" if youtube else "topic_draft")
    return ai_service.estimate_operations(operations)


def _validated_batch_selections(record: dict, selections: list[YoutubeBatchSelection]) -> list[dict]:
    seen, tasks = set(), []
    for selection in selections:
        concept = _video_concept(record, selection.concept_index)
        key = _slug(concept["canonical_name"])
        if key in seen:
            continue
        seen.add(key)
        existing = database.find_active_video_draft(record["id"], concept["canonical_name"])
        queue_item = database.get_queue_item_by_concept(record["id"], key)
        match = concept.get("existing_topic_match")
        if existing:
            tasks.append({"concept": concept, "action": selection.action, "state": "existing_draft", "draft_id": existing["id"]})
        elif match and selection.action != "enrich":
            raise HTTPException(422, f"{concept['canonical_name']} matches an existing topic and must use Enrich existing.")
        elif not match and selection.action != "create":
            raise HTTPException(422, f"{concept['canonical_name']} has no exact existing-topic match and must use Create new.")
        elif queue_item and queue_item["status"] in {"pending", "generating", "ready"}:
            tasks.append({"concept": concept, "action": selection.action, "state": queue_item["status"], "queue_id": queue_item["id"], "draft_id": queue_item.get("draft_id")})
        else:
            tasks.append({"concept": concept, "action": selection.action, "state": "new"})
    return tasks


@app.post("/api/youtube/imports/{import_id}/draft-queue/preflight")
def batch_preflight(import_id: str, body: YoutubeBatchInput):
    record = database.get_youtube_import(import_id)
    if not record:
        raise HTTPException(404, "Video import not found")
    tasks = _validated_batch_selections(record, body.selections)
    estimate_per = ai_service.estimate_operations(_topic_quality_operations("youtube_topic_expansion"))
    maximum_per = estimate_per["maximum_estimated_cost_usd"]
    billable = [task for task in tasks if task["state"] in {"new", "failed"}]
    maximum = maximum_per * len(billable)
    usage = ai_service.usage_summary()
    return {"tasks": [{"canonical_name": task["concept"]["canonical_name"], "action": task["action"], "state": task["state"],
        "draft_id": task.get("draft_id"), "queue_id": task.get("queue_id")} for task in tasks],
        "selected_count": len(tasks), "billable_count": len(billable), "maximum_estimated_cost_usd": maximum,
        "per_topic_operations": estimate_per["operations"], "remaining_budget_usd": usage["remaining_budget_usd"], "safe_to_start": maximum <= usage["remaining_budget_usd"]}


@app.post("/api/youtube/imports/{import_id}/draft-queue")
def enqueue_video_drafts(import_id: str, body: YoutubeBatchInput):
    record = database.get_youtube_import(import_id)
    if not record:
        raise HTTPException(404, "Video import not found")
    preflight = batch_preflight(import_id, body)
    if not preflight["safe_to_start"]:
        raise HTTPException(429, "The selected batch could exceed the local monthly AI budget. Reduce the selection or raise the budget intentionally in Settings.")
    tasks = _validated_batch_selections(record, body.selections)
    maximum = ai_service.estimate_operations(_topic_quality_operations("youtube_topic_expansion"))["maximum_estimated_cost_usd"]
    for task in tasks:
        key = _slug(task["concept"]["canonical_name"])
        if task["state"] == "existing_draft":
            database.create_queue_item(str(uuid4()), youtube_import_id=import_id, canonical_concept=key, concept=task["concept"],
                action=task["action"], status="ready", draft_id=task["draft_id"], maximum_cost_usd=maximum)
        elif task["state"] == "new":
            item = database.create_queue_item(str(uuid4()), youtube_import_id=import_id, canonical_concept=key, concept=task["concept"],
                action=task["action"], status="pending", maximum_cost_usd=maximum)
            if item["status"] == "failed":
                database.update_queue_item(item["id"], status="pending")
    return {"preflight": preflight, "queue": database.list_queue_items(import_id)}


@app.get("/api/youtube/imports/{import_id}/draft-queue")
def video_draft_queue(import_id: str):
    record = database.get_youtube_import(import_id)
    if not record:
        raise HTTPException(404, "Video import not found")
    _sync_video_drafts_to_queue(import_id, record.get("analysis"))
    return {"import_id": import_id, "queue": database.list_queue_items(import_id), "budget": ai_service.usage_summary()}


@app.get("/api/youtube/draft-queue")
def all_draft_queue():
    for record in database.list_youtube_imports():
        _sync_video_drafts_to_queue(record["id"], record.get("analysis"))
    return {"queue": database.list_queue_items(), "budget": ai_service.usage_summary()}


@app.post("/api/youtube/imports/{import_id}/analyze")
def analyze_youtube_import(import_id: str):
    record = database.get_youtube_import(import_id)
    if not record:
        raise HTTPException(404, "Video import not found")
    document = youtube_service.load_transcript(record["transcript_cache_path"])
    chunks = youtube_service.chunks(document)
    maximum = ai_service.estimate_maximum("youtube_concept_extraction") * len(chunks)
    if ai_service.usage_summary()["estimated_cost_usd"] + maximum > ai_service.settings().monthly_budget_usd:
        raise HTTPException(429, "This analysis would exceed the local monthly AI budget. The transcript remains available for later analysis.")
    candidates, arcs, actual_cost = [], [], 0.0
    for index, chunk in enumerate(chunks):
        request = {"video": document.source.model_dump(), "transcript_segment": chunk,
                   "instruction": "Extract only the few most useful concepts from this segment."}
        try:
            batch, _, cost = ai_service.generate(operation_type="youtube_concept_extraction",
                instructions=YOUTUBE_CONCEPT_INSTRUCTIONS, input_text=json.dumps(request),
                schema_name="ultimate_ml_youtube_concepts", schema=strict_response_schema(VideoConceptBatch),
                validate=VideoConceptBatch.model_validate, max_output_tokens=1600,
                metadata={"prompt_version": YOUTUBE_CONCEPT_PROMPT_VERSION, "youtube_import_id": import_id, "chunk_index": index})
        except Exception as exc:
            _ai_error(exc)
        candidates.extend(item.model_dump() for item in batch.concepts)
        arcs.append(batch.learning_arc)
        actual_cost += cost
    merged = {}
    rank = {"core": 3, "supporting": 2, "incidental": 1}
    for concept in candidates:
        key = _slug(concept["canonical_name"])
        existing = merged.get(key)
        if not existing or rank[concept["importance"]] > rank[existing["importance"]]:
            merged[key] = concept
        else:
            existing["timestamp_seconds"] = sorted(set(existing["timestamp_seconds"] + concept["timestamp_seconds"]))[:8]
    concepts = []
    for concept in sorted(merged.values(), key=lambda item: (-rank[item["importance"]], item["canonical_name"])):
        match = youtube_service.match_existing_topic(concept["canonical_name"], library.topics)
        concepts.append({**concept, "existing_topic_match": match,
            "allowed_actions": ["enrich", "ignore"] if match else ["create", "ignore"]})
    analysis = {"prompt_version": YOUTUBE_CONCEPT_PROMPT_VERSION, "learning_arc": " ".join(dict.fromkeys(arcs))[:1200], "concepts": concepts,
                "analysis_cost_estimated_usd": actual_cost, "source_vs_ai": {"source": "Concise evidence summaries and timestamps come from the local transcript.", "ai": "Concept ranking and learning-value phrasing are AI-generated review aids."}}
    database.update_youtube_import(import_id, state="analyzed", analysis=analysis)
    return _public_youtube_import(database.get_youtube_import(import_id))


@app.post("/api/youtube/imports/{import_id}/concepts/{concept_index}/expand")
def expand_youtube_concept(import_id: str, concept_index: int, body: YoutubeExpansionInput):
    record = database.get_youtube_import(import_id)
    if not record:
        raise HTTPException(404, "Analyze a video import before expanding a concept")
    return _generate_video_concept_draft(record, _video_concept(record, concept_index), body.action, body.focus)


def _generate_queue_item(item: dict) -> dict:
    record = database.get_youtube_import(item["youtube_import_id"])
    if not record:
        database.update_queue_item(item["id"], status="failed", error_message="Source video import no longer exists.")
        raise HTTPException(404, "Source video import no longer exists.")
    active = database.find_active_video_draft(record["id"], item["concept"]["canonical_name"])
    if active:
        database.update_queue_item(item["id"], status="ready", draft_id=active["id"], actual_cost_usd=0)
        return {"queue_item": database.get_queue_item(item["id"]), "draft": {"id": active["id"], "reused": True}, "budget": ai_service.usage_summary()}
    database.update_queue_item(item["id"], status="generating")
    try:
        draft = _generate_video_concept_draft(record, item["concept"], item["action"])
    except HTTPException as exc:
        message = str(exc.detail)
        database.update_queue_item(item["id"], status="failed", error_message=message)
        raise
    except Exception as exc:
        database.update_queue_item(item["id"], status="failed", error_message="Generation failed. Check Settings and retry this item.")
        raise HTTPException(502, "AI generation failed. Check the local API configuration and try again.") from exc
    database.update_queue_item(item["id"], status="ready", draft_id=draft["id"], actual_cost_usd=draft["usage"]["estimated_cost_usd"])
    return {"queue_item": database.get_queue_item(item["id"]), "draft": {"id": draft["id"], "reused": draft.get("reused", False), "usage": draft["usage"]}, "budget": ai_service.usage_summary()}


@app.post("/api/youtube/draft-queue/{item_id}/generate")
def generate_queue_item(item_id: str):
    item = database.get_queue_item(item_id)
    if not item:
        raise HTTPException(404, "Draft queue item not found")
    if item["status"] not in {"pending", "failed"}:
        return {"queue_item": item, "skipped": True, "budget": ai_service.usage_summary()}
    return _generate_queue_item(item)


@app.post("/api/youtube/imports/{import_id}/draft-queue/generate-next")
def generate_next_queue_item(import_id: str):
    item = database.next_pending_queue_item(import_id)
    if not item:
        return {"complete": True, "queue": database.list_queue_items(import_id), "budget": ai_service.usage_summary()}
    result = _generate_queue_item(item)
    return {"complete": False, **result, "queue": database.list_queue_items(import_id)}


@app.post("/api/youtube/imports/{import_id}/question-draft")
def create_youtube_question_draft(import_id: str, body: YoutubeQuestionInput):
    record = database.get_youtube_import(import_id)
    topic = library.topics.get(body.topic_id)
    if not record or not record["analysis"] or not topic:
        raise HTTPException(404, "Video import or approved topic not found")
    request = {"topic_id": topic["id"], "topic_title": topic["title"], "topic_summary": topic["one_sentence_summary"],
        "topic_core_explanation": topic["core_explanation"], "focus": body.focus, "count": min(12, max(1, body.count)),
        "source_context": record["analysis"], "known_question_text": [q["question"] for q in library.questions.values()]}
    try:
        batch, result, cost = ai_service.generate(operation_type="youtube_question_generation", instructions=QUESTION_INSTRUCTIONS,
            input_text=json.dumps(request), schema_name="ultimate_ml_question_draft", schema=strict_response_schema(QuestionDraftBatch),
            validate=QuestionDraftBatch.model_validate, max_output_tokens=2600,
            metadata={"prompt_version": QUESTION_PROMPT_VERSION, "topic_id": body.topic_id, "youtube_import_id": import_id})
    except Exception as exc:
        _ai_error(exc)
    payload = {"topic_id": body.topic_id, "approval_state": "draft", "questions": [{**item.model_dump(), "selected": True, "approval_status": "selected"} for item in batch.questions],
        "generation_metadata": {"generated_by_ai": True, "provider": "openai", "model": ai_service.settings().model,
            "prompt_version": QUESTION_PROMPT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
            "user_focus": body.focus, "review_state": "draft", "youtube_import_id": import_id}}
    draft_id = str(uuid4())
    database.create_draft(draft_id, "question", f"Questions for {topic['title']}", payload, {"request_cost_usd": cost, "youtube_import_id": import_id})
    return {"id": draft_id, "state": "draft", "payload": payload, "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "estimated_cost_usd": cost}}


@app.post("/api/ai/topic-draft")
def create_topic_draft(body: TopicDraftInput):
    return _generate_topic_draft(body)


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
    payload = {"topic_id": body.topic_id, "approval_state": "draft", "questions": [{**item.model_dump(), "selected": True, "approval_status": "selected"} for item in batch.questions],
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
    if draft["draft_type"] == "question":
        draft["payload"] = _question_candidate_status(draft["payload"])
    return _draft_public_quality(draft) if draft["draft_type"] == "topic" else draft


@app.get("/api/ai/topic-drafts")
def list_topic_drafts():
    """A cross-source queue view; deriving legacy status performs no writes."""
    return {"drafts": [_draft_public_quality(item) for item in database.list_drafts_by_type("topic") if item["state"] == "draft"],
            "budget": ai_service.usage_summary()}


@app.get("/api/ai/drafts/{draft_id}/quality-review-estimate")
def existing_draft_quality_review_estimate(draft_id: str):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    if draft["draft_type"] != "topic" or draft["state"] != "draft":
        raise HTTPException(409, "Only active topic drafts can be quality-reviewed.")
    estimate = ai_service.estimate_operations(["topic_quality_review_existing", "metadata_relationship_resolution"])
    return {**estimate, "quality_review": _draft_quality_state(draft["payload"]),
        "original_generation_estimated_cost_usd": draft["metadata"].get("request_cost_usd"),
        "reviewer_prompt_version": TOPIC_QUALITY_REVIEW_PROMPT_VERSION,
        "original_generation_will_not_repeat": True}


@app.post("/api/ai/drafts/{draft_id}/quality-review")
def quality_review_existing_draft(draft_id: str, body: ExistingDraftQualityReviewInput):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    return _run_existing_draft_quality_review(draft, force=body.force)


@app.get("/api/ai/drafts/{draft_id}/metadata-rebuild-estimate")
def draft_metadata_rebuild_estimate(draft_id: str):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    if draft["draft_type"] != "topic" or draft["state"] != "draft":
        raise HTTPException(409, "Only active topic drafts can rebuild metadata.")
    return _metadata_rebuild_estimate(draft["payload"])


@app.post("/api/ai/drafts/{draft_id}/rebuild-relationships")
def rebuild_draft_relationships(draft_id: str):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    if draft["draft_type"] != "topic" or draft["state"] != "draft":
        raise HTTPException(409, "Only active topic drafts can rebuild metadata.")
    original = json.loads(json.dumps(draft["payload"]))
    estimate = _metadata_rebuild_estimate(original)
    if not estimate["cached"]:
        try:
            ai_service.require_budget_for_operations(["metadata_relationship_resolution"])
        except Exception as exc:
            _ai_error(exc)
    rebuilt, _, cost, cached, warnings, blocking = _resolve_topic_metadata(original, assigned_topic_id=_slug(draft["title"]),
        metadata={"draft_id": draft_id, "metadata_only_rebuild": True})
    report = rebuilt.get("quality_report")
    if isinstance(report, dict):
        report = final_quality_report(report, warnings, blocking, payload=rebuilt)
        report["blocking_issues_remaining"] += metadata_payload_consistency(rebuilt, report, catalog=_topic_catalog(), assigned_topic_id=_slug(draft["title"]))
        rebuilt["quality_report"] = report
        rebuilt["quality_status"] = "needs_attention" if report["blocking_issues_remaining"] else "ready"
    database.create_draft_quality_revision(str(uuid4()), draft_id, "metadata_rebuild", original,
        source_payload_hash=_payload_hash(original), reviewer_prompt_version=RELATIONSHIP_RESOLVER_PROMPT_VERSION)
    database.update_draft(draft_id, rebuilt)
    return {"draft": _draft_public_quality(database.get_draft(draft_id)), "usage": {"estimated_cost_usd": cost,
            "maximum_estimated_cost_usd": estimate["maximum_estimated_cost_usd"], "cached": cached}, "warnings": warnings,
            "budget": ai_service.usage_summary()}


@app.get("/api/ai/drafts/{draft_id}/quality-revisions")
def quality_revisions(draft_id: str):
    if not database.get_draft(draft_id):
        raise HTTPException(404, "Draft not found")
    revisions = database.list_draft_quality_revisions(draft_id)
    return {"revisions": [{key: value for key, value in item.items() if key != "payload"} for item in revisions]}


@app.get("/api/ai/drafts/{draft_id}/quality-revisions/{revision_id}")
def quality_revision(draft_id: str, revision_id: str):
    revision = database.get_draft_quality_revision(draft_id, revision_id)
    if not revision:
        raise HTTPException(404, "Quality revision not found")
    return revision


@app.post("/api/ai/drafts/{draft_id}/quality-revisions/{revision_id}/restore")
def restore_quality_revision(draft_id: str, revision_id: str):
    draft = database.get_draft(draft_id)
    revision = database.get_draft_quality_revision(draft_id, revision_id)
    if not draft or not revision:
        raise HTTPException(404, "Draft or quality revision not found")
    if draft["draft_type"] != "topic" or draft["state"] != "draft":
        raise HTTPException(409, "Only active topic drafts can be restored.")
    database.update_draft(draft_id, revision["payload"])
    return _draft_public_quality(database.get_draft(draft_id))


@app.put("/api/ai/drafts/{draft_id}")
def update_draft(draft_id: str, body: DraftPayloadInput):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    payload = body.payload
    if draft["draft_type"] == "topic":
        # The normal editor only offers known topics. Keep the advanced editor
        # safe as well: unavailable IDs never persist as relationships.
        payload = _sanitize_draft_relationships(payload)
    database.update_draft(draft_id, payload)
    return database.get_draft(draft_id)


@app.get("/api/ai/drafts/{draft_id}/validate")
def validate_draft(draft_id: str):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    if draft["draft_type"] != "topic":
        return {"valid": True, "errors": [], "warnings": []}
    return _validate_topic_draft_payload(_sanitize_draft_relationships(draft["payload"]))


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
    database.update_queue_item_for_draft(draft_id, "discarded")
    return database.get_draft(draft_id)


@app.post("/api/ai/drafts/{draft_id}/approve")
def approve_draft(draft_id: str):
    draft = database.get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")
    if draft["draft_type"] == "topic" and draft["state"] != "draft":
        raise HTTPException(409, "Only an active draft can be approved")
    if draft["draft_type"] == "topic":
        validation = _validate_topic_draft_payload(_sanitize_draft_relationships(draft["payload"]))
        if not validation["valid"]:
            raise HTTPException(422, {"message": "Fix draft validation errors before approval.", **validation})
        enrichment_target_id = draft["payload"].get("generation_metadata", {}).get("enrich_existing_topic_id")
        topic = _approved_topic(draft["payload"], canonical_id=enrichment_target_id)
        if enrichment_target_id:
            existing = library.topics.get(enrichment_target_id)
            if not existing:
                raise HTTPException(422, "The topic selected for enrichment no longer exists.")
            topic["id"] = enrichment_target_id
            topic["sources"] = list({json.dumps(item, sort_keys=True): item for item in (existing.get("sources", []) + topic.get("sources", []))}.values())
            target = ROOT / existing["_path"]
        else:
            target = ROOT / "content" / "topics" / f"{topic['id']}.json"
        if target.exists() and not enrichment_target_id:
            raise HTTPException(409, "A topic with this ID already exists. Edit the draft title or use the existing topic.")
        _write_json(target, topic)
        try:
            _reload_library()
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(422, f"Draft cannot be approved: {exc}") from exc
        database.update_draft(draft_id, topic, "approved")
        database.update_queue_item_for_draft(draft_id, "approved")
        return {"state": "approved", "topic": public(library.topics[topic["id"]])}
    # Approving generated candidates is deliberately local: no provider call is reachable here.
    payload = _question_candidate_status(draft["payload"])
    topic_id = payload.get("topic_id")
    selected = [(index, item) for index, item in enumerate(payload.get("questions", [])) if item.get("selected")]
    logger.info("question_approval_start draft_id=%s selected_count=%s prior_state=%s", draft_id, len(selected), draft["state"])
    if not topic_id or topic_id not in library.topics or not selected:
        logger.warning("question_approval_validation draft_id=%s valid=false reason=missing_topic_or_selection", draft_id)
        raise HTTPException(422, "Select at least one question for an existing topic before approval")
    database.update_draft(draft_id, payload, "approving")
    prepared, validation_errors = [], []
    for index, candidate in selected:
        normalized, errors = _validate_question_candidate(candidate, topic_id)
        if errors:
            validation_errors.extend({"index": index, "question": candidate.get("question", ""), "reason": error} for error in errors)
        else:
            prepared.append((index, normalized))
    if validation_errors:
        payload["approval_state"] = "failed"
        database.update_draft(draft_id, payload, "failed")
        logger.warning("question_approval_validation draft_id=%s valid=false failed_count=%s", draft_id, len(validation_errors))
        raise HTTPException(422, {"message": "Question approval failed validation.", "failed": validation_errors})
    existing = [question for question in library.questions.values() if _question_matches_topic(question, topic_id)]
    existing_by_text = {_slug(question["question"]): question for question in existing}
    approved, already_approved, failures, reserved = [], [], [], set(library.questions)
    staged = []
    for index, candidate in prepared:
        payload["questions"][index].update(candidate)
        duplicate = existing_by_text.get(_slug(candidate["question"]))
        if duplicate:
            payload["questions"][index].update({"approval_status": "approved", "approved_question_id": duplicate["id"]})
            already_approved.append({"index": index, "id": duplicate["id"], "question": candidate["question"]})
            continue
        question_id = _next_question_id(topic_id, reserved)
        reserved.add(question_id)
        question = {"content_version": 2, "id": question_id, "topic_ids": [topic_id], "question_type": "conceptual",
            **candidate, "concept_refresher_topic_id": topic_id,
            "generation_metadata": {**payload.get("generation_metadata", {}), "review_state": "approved"}}
        staged.append((index, question))
    written = []
    try:
        for index, question in staged:
            target = ROOT / "content" / "questions" / f"{question['id']}.json"
            if target.exists():
                failures.append({"index": index, "reason": f"Question ID already exists: {question['id']}"})
                continue
            _atomic_write_json(target, question)
            written.append(target)
            payload["questions"][index].update({"approval_status": "approved", "approved_question_id": question["id"]})
            approved.append({"index": index, "id": question["id"], "question": question["question"]})
        if failures:
            raise RuntimeError("One or more question IDs already existed")
        _reload_library()
    except Exception as exc:
        # Keep successfully written files recoverable; the persisted per-item state makes a retry idempotent.
        payload["approval_state"] = "failed"
        database.update_draft(draft_id, payload, "failed")
        logger.exception("question_approval_failure draft_id=%s approved_count=%s duplicate_count=%s failed_count=%s exception_type=%s", draft_id, len(approved), len(already_approved), len(failures), type(exc).__name__)
        raise HTTPException(500, {"message": "Question approval did not complete. Check draft status and retry safely.", "approved": approved,
                                  "already_approved": already_approved, "failed": failures or [{"reason": str(exc)}]}) from exc
    payload["approval_state"] = "approved"
    database.update_draft(draft_id, payload, "approved")
    result = {"state": "approved", "approved": approved, "already_approved": already_approved, "failed": failures,
              "selected_count": len(selected), "message": f"{len(approved)} questions approved; {len(already_approved)} already approved."}
    logger.info("question_approval_complete draft_id=%s approved_count=%s duplicate_count=%s failed_count=%s status=200", draft_id, len(approved), len(already_approved), len(failures))
    return result


app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
