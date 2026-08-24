"""Deterministic safeguards around the model-assisted topic quality review."""
from __future__ import annotations

from copy import deepcopy
import re
from urllib.parse import urlparse

from ..youtube.transcript_provider import TranscriptUnavailableError, canonical_youtube_url


TAXONOMY = {
    "mathematical_foundations": {"definition": "Mathematical ideas used to reason about ML.", "examples": ["Covariance", "Linear Algebra"], "not_examples": ["CLIP", "Knowledge Distillation"]},
    "ml_fundamentals": {"definition": "Cross-family ML principles, not a particular neural method.", "examples": ["Overfitting", "Bias-Variance Tradeoff"], "not_examples": ["Knowledge Distillation", "CLIP", "ResNet"]},
    "classical_ml": {"definition": "Non-neural statistical or algorithmic ML methods.", "examples": ["PCA", "Gaussian Mixture Models"], "not_examples": ["DINO", "CLIP"]},
    "deep_learning": {"definition": "Neural-network architectures, objectives, and training methods.", "examples": ["Knowledge Distillation", "ResNet", "DINO"], "not_examples": ["Covariance"]},
    "computer_vision": {"definition": "General visual perception tasks and techniques.", "examples": ["Image Segmentation", "Visual Features"], "not_examples": ["Text-only language modelling"]},
    "transformers": {"definition": "Transformer architectures and attention-centric methods.", "examples": ["Self-Attention", "Vision Transformer"], "not_examples": ["PCA"]},
    "representation_learning": {"definition": "Learning useful embeddings and representation objectives, including multimodal contrastive methods.", "examples": ["Contrastive Learning", "CLIP", "SimCLR"], "not_examples": ["Gradient Descent"]},
    "object_detection": {"definition": "Object localisation and detection systems.", "examples": ["Region Proposal", "Detection Metrics"], "not_examples": ["Knowledge Distillation"]},
    "anomaly_detection": {"definition": "Detecting unusual observations or behaviour.", "examples": ["One-Class Classification"], "not_examples": ["CLIP"]},
    "uncertainty_calibration": {"definition": "Prediction uncertainty and calibration.", "examples": ["Expected Calibration Error"], "not_examples": ["ResNet"]},
    "research_evaluation": {"definition": "Experimental design, benchmarking, and research evaluation.", "examples": ["Ablation Studies"], "not_examples": ["Contrastive Learning"]},
}

CONCEPT_TYPES = ("broad_concept", "named_method", "architecture", "loss_or_objective", "mathematical_concept", "training_mechanism", "evaluation_concept")

# These are deliberately conservative guards for recurring real failures. They do not
# manufacture alternatives: an empty edge list is the safe outcome.
SPECIAL_TOPIC_POLICY = {
    "knowledge-distillation": {"category": "deep_learning", "concept_type": "training_mechanism",
        "forbidden_relations": {"gradient-descent", "cnn", "resnet", "gradient-flow", "self-supervised-learning"}},
    "clip": {"category": "representation_learning", "concept_type": "named_method",
        "forbidden_relations": {"covariance", "gradient-descent", "cnn", "resnet", "gradient-flow"}},
    "contrastive-learning": {"category": "representation_learning", "concept_type": "loss_or_objective",
        "forbidden_relations": {"gradient-descent", "cnn", "pca", "resnet", "gradient-flow"}},
}


def taxonomy_context() -> dict:
    return {key: value for key, value in TAXONOMY.items()}


def _slug(value: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", value.casefold()))


def _issue(area: str, message: str) -> dict:
    return {"area": area, "message": message}


def _canonical_url(value: str | None) -> str | None:
    if not value:
        return value
    host = urlparse(value).netloc.casefold().split(":")[0]
    if host in {"youtu.be", "www.youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com"}:
        try:
            return canonical_youtube_url(value)
        except TranscriptUnavailableError:
            return value
    return value


def _equation_issues(payload: dict) -> list[dict]:
    issues = []
    foundation = payload.get("mathematical_foundation") or {}
    for section_index, section in enumerate(foundation.get("sections", [])):
        for equation_index, equation in enumerate(section.get("equations", [])):
            latex = str(equation.get("latex", "")).strip()
            explanation = str(equation.get("explanation", "")).strip()
            label = f"Equation {section_index + 1}.{equation_index + 1}"
            if not latex or not explanation:
                issues.append(_issue("mathematics", f"{label} needs non-empty LaTeX and an explanation."))
            elif latex.count("{") != latex.count("}") or latex.endswith("\\"):
                issues.append(_issue("mathematics", f"{label} has malformed LaTeX delimiters."))
            elif re.search(r"[A-Za-z0-9]\"", latex):
                issues.append(_issue("mathematics", f"{label} contains suspicious quote notation that needs review."))
    return issues


def normalize_topic(payload: dict, *, requested_title: str, assigned_topic_id: str, catalog: list[dict], source_context: dict | None = None,
                    enforce_justifications: bool = True) -> tuple[dict, list[dict], list[dict]]:
    """Normalize only generated content; never mutate an approved file or an old draft."""
    topic = deepcopy(payload)
    warnings, blocking = [], []
    topic.pop("id", None)  # A model ID is never durable identity.
    topic["title"] = requested_title or topic.get("title", "")
    topic["durable_topic_id"] = assigned_topic_id  # draft-only, backend-owned preview identity
    topic["tags"] = list(dict.fromkeys(tag.strip() for tag in topic.get("tags", []) if tag and tag.strip()))
    policy = SPECIAL_TOPIC_POLICY.get(_slug(requested_title))
    if topic.get("category") not in TAXONOMY:
        blocking.append(_issue("taxonomy", f"Unknown category '{topic.get('category')}'."))
    if policy and topic.get("category") != policy["category"]:
        topic["category"] = policy["category"]
        warnings.append(_issue("taxonomy", f"Applied the taxonomy policy for {requested_title}."))
    if topic.get("concept_type") not in CONCEPT_TYPES:
        topic["concept_type"] = "broad_concept"
        warnings.append(_issue("taxonomy", "Unrecognized concept type was normalized to broad_concept."))
    if policy:
        topic["concept_type"] = policy["concept_type"]

    catalog_ids = {item["id"] for item in catalog}
    justifications = {}
    for relation in topic.get("relationship_justifications", []):
        if isinstance(relation, dict) and relation.get("topic_id") in catalog_ids:
            justifications[(relation.get("relationship"), relation.get("topic_id"))] = relation
    clean_justifications = []
    used = set()
    for field, kind in (("prerequisite_topic_ids", "prerequisite"), ("related_topic_ids", "related")):
        clean = []
        for topic_id in topic.get(field, []):
            relation = justifications.get((kind, topic_id))
            forbidden = policy and topic_id in policy["forbidden_relations"]
            if topic_id not in catalog_ids:
                warnings.append(_issue("relationships", f"Removed unavailable {kind} ID '{topic_id}'."))
            elif topic_id == assigned_topic_id:
                warnings.append(_issue("relationships", "Removed a self relationship."))
            elif topic_id in used:
                warnings.append(_issue("relationships", f"Removed {topic_id} because a topic cannot be both prerequisite and related."))
            elif forbidden:
                warnings.append(_issue("relationships", f"Removed weak {kind} relationship '{topic_id}' for this concept."))
            elif enforce_justifications and not relation:
                warnings.append(_issue("relationships", f"Removed {kind} relationship '{topic_id}' without an explicit rationale."))
            else:
                clean.append(topic_id); used.add(topic_id)
                if relation:
                    clean_justifications.append(relation)
        topic[field] = list(dict.fromkeys(clean))
    topic["relationship_justifications"] = clean_justifications
    suggestions, suggested_keys = [], set()
    for item in topic.get("suggested_new_topic_relationships", []):
        if not isinstance(item, dict):
            continue
        key = (_slug(str(item.get("title", ""))), item.get("relationship"))
        if key[0] and key not in suggested_keys:
            suggestions.append(item); suggested_keys.add(key)
    topic["suggested_new_topic_relationships"] = suggestions

    for source in topic.get("sources", []):
        if isinstance(source, dict):
            source["url"] = _canonical_url(source.get("url"))
    provenance = topic.get("source_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("source_derived"), dict):
        source_derived = provenance["source_derived"]
        source_derived["video_url"] = _canonical_url(source_derived.get("video_url"))
    if source_context:
        # The server, not model output, supplies provenance for video-derived drafts.
        canonical = _canonical_url(source_context.get("video_url"))
        topic["sources"] = [{"title": source_context["title"], "type": "youtube", "url": canonical}]
        provenance_keys = ("kind", "video_id", "title", "channel", "attribution", "concept", "source_evidence_summary", "timestamp_seconds")
        topic["source_provenance"] = {"source_derived": {key: source_context.get(key) for key in provenance_keys} | {"video_url": canonical},
            "ai_expanded": "The educational explanation was AI-expanded from the selected source concept and requires human review."}
    blocking.extend(_equation_issues(topic))
    topic["relationship_warnings"] = warnings
    return topic, warnings, blocking


def final_quality_report(reviewer_report: dict, deterministic_warnings: list[dict], deterministic_blocking: list[dict]) -> dict:
    report = deepcopy(reviewer_report)
    report.setdefault("blocking_issues_fixed", [])
    report.setdefault("blocking_issues_remaining", [])
    report.setdefault("warnings", [])
    report.setdefault("confidence", "medium")
    report["warnings"] = report["warnings"] + deterministic_warnings
    report["blocking_issues_remaining"] = report["blocking_issues_remaining"] + deterministic_blocking
    return report
