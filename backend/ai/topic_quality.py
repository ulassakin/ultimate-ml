"""Deterministic safeguards for educational topic quality, not a topic graph."""
from __future__ import annotations

from copy import deepcopy
import re
from urllib.parse import urlparse

from ..youtube.transcript_provider import TranscriptUnavailableError, canonical_youtube_url, youtube_video_id


CONCEPT_TYPES = (
    "broad_concept", "named_method", "architecture", "loss_or_objective",
    "mathematical_concept", "training_mechanism", "evaluation_concept",
)
TAXONOMY_REGISTRY = {
    "mathematical_foundations": {"display_name": "Mathematical Foundations", "definition": "Mathematical concepts needed to reason about ML.", "positive_examples": ["Covariance", "Eigenvalues"], "negative_examples": ["CLIP", "ResNet"], "compatible_concept_types": ["mathematical_concept"]},
    "ml_fundamentals": {"display_name": "ML Fundamentals", "definition": "Cross-family ML principles rather than a particular method.", "positive_examples": ["Overfitting", "Bias-Variance"], "negative_examples": ["Knowledge Distillation", "CLIP"], "compatible_concept_types": ["broad_concept", "evaluation_concept"]},
    "classical_ml": {"display_name": "Classical ML", "definition": "Non-neural statistical or algorithmic ML methods.", "positive_examples": ["PCA", "Gaussian Mixture Models"], "negative_examples": ["DINO", "Vision Transformer"], "compatible_concept_types": ["broad_concept", "mathematical_concept", "loss_or_objective"]},
    "deep_learning": {"display_name": "Deep Learning", "definition": "Neural architectures, objectives, and training mechanisms.", "positive_examples": ["Knowledge Distillation", "ResNet", "DINO"], "negative_examples": ["Covariance"], "compatible_concept_types": ["architecture", "named_method", "training_mechanism", "loss_or_objective", "broad_concept"]},
    "computer_vision": {"display_name": "Computer Vision", "definition": "Visual perception tasks and techniques.", "positive_examples": ["Image Segmentation"], "negative_examples": ["Text-only language modeling"], "compatible_concept_types": ["broad_concept", "evaluation_concept"]},
    "transformers": {"display_name": "Transformers", "definition": "Attention and transformer architectures.", "positive_examples": ["Self-Attention", "Vision Transformer"], "negative_examples": ["PCA"], "compatible_concept_types": ["architecture", "named_method", "broad_concept"]},
    "representation_learning": {"display_name": "Representation Learning", "definition": "Learning embeddings and representation objectives, including multimodal contrastive methods.", "positive_examples": ["Contrastive Learning", "CLIP", "SimCLR"], "negative_examples": ["Gradient Descent"], "compatible_concept_types": ["named_method", "loss_or_objective", "broad_concept", "training_mechanism"]},
    "object_detection": {"display_name": "Object Detection", "definition": "Object localization and detection systems.", "positive_examples": ["Region Proposal"], "negative_examples": ["Knowledge Distillation"], "compatible_concept_types": ["broad_concept", "architecture", "evaluation_concept"]},
    "anomaly_detection": {"display_name": "Anomaly Detection", "definition": "Detecting unusual observations or behavior.", "positive_examples": ["One-Class Classification"], "negative_examples": ["CLIP"], "compatible_concept_types": ["broad_concept", "evaluation_concept"]},
    "uncertainty_calibration": {"display_name": "Uncertainty & Calibration", "definition": "Prediction uncertainty and calibration.", "positive_examples": ["Expected Calibration Error"], "negative_examples": ["ResNet"], "compatible_concept_types": ["broad_concept", "evaluation_concept", "mathematical_concept"]},
    "research_evaluation": {"display_name": "Research Evaluation", "definition": "Experimental design, benchmarks, and research evaluation.", "positive_examples": ["Ablation Studies"], "negative_examples": ["Contrastive Learning"], "compatible_concept_types": ["evaluation_concept", "broad_concept"]},
}
MATH_VOCABULARY = {
    "vectors": "Vectors and matrices", "matrix": "Vectors and matrices", "dot": "Dot products", "norm": "Norms",
    "cos": "Cosine similarity", "softmax": "Softmax", "cross-entropy": "Cross-entropy", "kl": "KL divergence",
    "entropy": "Entropy", "temperature": "Temperature scaling", "expectation": "Expectations", "variance": "Variance",
    "covariance": "Covariance", "eigen": "Eigenvalues and eigenvectors", "gradient": "Gradients", "chain rule": "Chain rule",
    "contrastive": "Contrastive learning intuition",
}
ARCHITECTURE_MATH_TERMS = {"cnn", "convolutional neural network", "resnet", "transformer", "vision transformer", "gradient descent", "gradient-descent"}
LEGACY_RELATIONSHIP_FIELDS = {
    "prerequisite_topic_ids", "related_topic_ids", "relationship_justifications",
    "suggested_new_topic_relationships", "relationship_warnings", "metadata_resolution",
}


def taxonomy_context() -> dict:
    return TAXONOMY_REGISTRY


def _taxonomy_token(value: object) -> str:
    """Make labels and internal IDs comparable without accepting new categories."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold())).strip("_")


def canonical_category(value: object) -> str:
    """Return the authoritative taxonomy ID for a known label, otherwise retain it.

    The registry is deliberately the only source of aliases: every category ID
    and its human display label are accepted case-insensitively, with spaces,
    hyphens, ampersands, and underscores treated equivalently.  An unknown
    value is *not* guessed, so the normal taxonomy blocker remains useful.
    """
    token = _taxonomy_token(value)
    for category_id, definition in TAXONOMY_REGISTRY.items():
        if token in {_taxonomy_token(category_id), _taxonomy_token(definition["display_name"])}:
            return category_id
    return str(value or "").strip()


def _slug(value: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", value.casefold()))


def _issue(area: str, message: str) -> dict:
    return {"area": area, "message": message}


def _dedupe_issues(items: list[dict]) -> list[dict]:
    result, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("area", "")).casefold(), re.sub(r"\s+", " ", str(item.get("message", "")).strip()).casefold())
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


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
    for section_index, section in enumerate((payload.get("mathematical_foundation") or {}).get("sections", [])):
        for equation_index, equation in enumerate(section.get("equations", []) if isinstance(section, dict) else []):
            latex, explanation = str(equation.get("latex", "")).strip(), str(equation.get("explanation", "")).strip()
            label = f"Equation {section_index + 1}.{equation_index + 1}"
            if not latex or not explanation:
                issues.append(_issue("mathematics", f"{label} needs non-empty LaTeX and an explanation."))
            elif latex.count("{") != latex.count("}") or latex.endswith("\\"):
                issues.append(_issue("mathematics", f"{label} has malformed LaTeX delimiters."))
    return issues


def _normalize_math_prerequisites(topic: dict, warnings: list[dict]) -> None:
    """Optional display hints; they never become graph data or approval blockers."""
    foundation = topic.get("mathematical_foundation")
    if not isinstance(foundation, dict):
        return
    allowed = {value.casefold(): value for value in MATH_VOCABULARY.values()}
    def clean(items: object, label: str) -> list[str]:
        result = []
        for value in items if isinstance(items, list) else []:
            if not isinstance(value, str) or not value.strip():
                continue
            if value.casefold() in ARCHITECTURE_MATH_TERMS:
                warnings.append(_issue("mathematics", f"Removed architecture/training item '{value}' from {label} mathematical background hints."))
                continue
            canonical = allowed.get(value.casefold())
            if canonical:
                result.append(canonical)
        return list(dict.fromkeys(result))
    foundation["prerequisites"] = clean(foundation.get("prerequisites", []), "global")
    for section in foundation.get("sections", []):
        if isinstance(section, dict):
            section["prerequisites"] = clean(section.get("prerequisites", []), f"section '{section.get('title', 'untitled')}'")


def _source_grounding_checks(topic: dict) -> tuple[list[dict], list[dict]]:
    warnings, blocking = [], []
    sources, provenance = topic.get("sources", []), topic.get("source_provenance")
    if not (provenance or any(isinstance(source, dict) and source.get("type") == "youtube" for source in sources)):
        return warnings, blocking
    if not isinstance(provenance, dict) or not isinstance(provenance.get("source_derived"), dict) or not isinstance(provenance.get("ai_expanded"), str):
        return warnings, [_issue("provenance", "Video-derived content needs clear source_derived and AI-expanded provenance.")]
    source = provenance["source_derived"]
    if not source.get("source_evidence_summary"):
        blocking.append(_issue("provenance", "Source-derived evidence needs a summary."))
    if source.get("kind") == "youtube":
        if not source.get("video_url"):
            blocking.append(_issue("provenance", "YouTube source provenance needs a canonical video URL."))
        else:
            try:
                url_id = youtube_video_id(source["video_url"])
                if source.get("video_id") and source["video_id"] != url_id:
                    blocking.append(_issue("source_grounding", "Source provenance video ID contradicts its canonical URL."))
            except TranscriptUnavailableError:
                blocking.append(_issue("provenance", "Source provenance does not contain a valid canonical YouTube URL."))
    timestamps = source.get("timestamp_seconds")
    if timestamps is not None and (not isinstance(timestamps, list) or any(not isinstance(item, (int, float)) or item < 0 for item in timestamps)):
        blocking.append(_issue("source_grounding", "Source evidence timestamps are malformed or contradictory."))
    elif isinstance(timestamps, list):
        claimed = [int(value) for value in re.findall(r"(?:at|around|~)\s*(\d+)\s*s\b", str(source.get("source_evidence_summary", "")).casefold())]
        if claimed and not any(abs(value - timestamp) <= 30 for value in claimed for timestamp in timestamps):
            blocking.append(_issue("source_grounding", "Source evidence summary timestamp contradicts structured provenance timestamps."))
    if len(str(source.get("source_evidence_summary", "")).split()) <= 40 and "ai-expand" in provenance.get("ai_expanded", "").casefold():
        warnings.append(_issue("source_grounding", "Source segment is brief; the broader explanation is explicitly marked AI-expanded."))
    return warnings, blocking


def _method_completeness_checks(topic: dict) -> tuple[list[dict], list[dict]]:
    warnings, blocking = [], []
    if _slug(topic.get("title", "")) != "simclr":
        return warnings, blocking
    text = " ".join(str(topic.get(field, "")) for field in ("core_explanation", "mechanism", "quick_recall")).casefold()
    equations = " ".join(equation.get("latex", "") for section in (topic.get("mathematical_foundation") or {}).get("sections", []) for equation in section.get("equations", []) if isinstance(section, dict))
    if not all(token in text for token in ("augment", "projection", "contrast")):
        blocking.append(_issue("named_method_completeness", "SimCLR is missing a core-defining augmentation, projection-head, or contrastive-loss mechanism."))
    if "ell_{j,i}" not in equations and not any(marker in text for marker in ("i→j", "j→i", "i->j", "j->i")):
        warnings.append(_issue("named_method_completeness", "SimCLR does not explicitly state symmetric i→j and j→i NT-Xent averaging; this is a useful completeness detail, not a blocker when the loss is otherwise correct."))
    return warnings, blocking


def normalize_topic(payload: dict, *, requested_title: str, assigned_topic_id: str, catalog: list[dict] | None = None,
                    source_context: dict | None = None, enforce_justifications: bool = True) -> tuple[dict, list[dict], list[dict]]:
    """Normalize generated/reviewed educational content and discard graph-era output."""
    topic, warnings, blocking = deepcopy(payload), [], []
    for field in LEGACY_RELATIONSHIP_FIELDS:
        topic.pop(field, None)
    topic.pop("id", None)
    topic["title"] = requested_title or topic.get("title", "")
    topic["durable_topic_id"] = assigned_topic_id
    # This is intentionally before taxonomy and concept-type checks. Models and
    # human editors may use the display label, but persistence always uses the
    # canonical internal taxonomy ID.
    topic["category"] = canonical_category(topic.get("category"))
    topic["tags"] = list(dict.fromkeys(tag.strip() for tag in topic.get("tags", []) if isinstance(tag, str) and tag.strip()))
    if topic.get("category") not in TAXONOMY_REGISTRY:
        blocking.append(_issue("taxonomy", f"Unknown category '{topic.get('category')}'."))
    if topic.get("concept_type") not in CONCEPT_TYPES:
        topic["concept_type"] = "broad_concept"
        warnings.append(_issue("taxonomy", "Unrecognized concept type was normalized to broad_concept."))
    elif topic["concept_type"] not in TAXONOMY_REGISTRY.get(topic.get("category"), {}).get("compatible_concept_types", []):
        blocking.append(_issue("taxonomy", f"Concept type '{topic['concept_type']}' is not compatible with category '{topic.get('category')}'."))
    for source in topic.get("sources", []):
        if isinstance(source, dict):
            source["url"] = _canonical_url(source.get("url"))
    if source_context:
        canonical = _canonical_url(source_context.get("video_url"))
        keys = ("kind", "video_id", "title", "channel", "attribution", "concept", "source_evidence_summary", "timestamp_seconds")
        topic["sources"] = [{"title": source_context["title"], "type": "youtube", "url": canonical}]
        topic["source_provenance"] = {"source_derived": {key: source_context.get(key) for key in keys} | {"video_url": canonical}, "ai_expanded": "The educational explanation was AI-expanded from the selected source concept and requires human review."}
    provenance = topic.get("source_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("source_derived"), dict):
        provenance["source_derived"]["video_url"] = _canonical_url(provenance["source_derived"].get("video_url"))
    _normalize_math_prerequisites(topic, warnings)
    source_warnings, source_blocking = _source_grounding_checks(topic)
    method_warnings, method_blocking = _method_completeness_checks(topic)
    warnings.extend(source_warnings + method_warnings)
    blocking.extend(source_blocking + method_blocking + _equation_issues(topic))
    return topic, _dedupe_issues(warnings), _dedupe_issues(blocking)


def final_quality_report(reviewer_report: dict, deterministic_warnings: list[dict], deterministic_blocking: list[dict], *, payload: dict | None = None) -> dict:
    report = deepcopy(reviewer_report)
    report.setdefault("blocking_issues_fixed", [])
    report.setdefault("blocking_issues_remaining", [])
    report.setdefault("warnings", [])
    report.setdefault("confidence", "medium")
    report["warnings"] += deterministic_warnings
    report["blocking_issues_remaining"] += deterministic_blocking
    if payload:
        category = payload.get("category")
        concept_type = payload.get("concept_type")
        category_is_valid = category in TAXONOMY_REGISTRY
        concept_is_compatible = category_is_valid and concept_type in TAXONOMY_REGISTRY[category].get("compatible_concept_types", [])
        # A reviewer can correctly repair a display label in corrected_topic
        # while accidentally retaining a criticism of its pre-normalized value
        # in the report. Keep genuine taxonomy feedback, but drop only those
        # stale blockers whose sole cause is now-resolved canonicalization.
        if category_is_valid and concept_is_compatible:
            stale_taxonomy_markers = ("unknown category", "not compatible with category", "incompatible with category", "canonical taxonomy id")
            report["blocking_issues_remaining"] = [
                issue for issue in report["blocking_issues_remaining"]
                if not (issue.get("area") == "taxonomy" and any(marker in str(issue.get("message", "")).casefold() for marker in stale_taxonomy_markers))
            ]
        provenance = payload.get("source_provenance", {})
        honest_expansion = isinstance(provenance, dict) and isinstance(provenance.get("source_derived"), dict) and "ai-expand" in str(provenance.get("ai_expanded", "")).casefold()
        if honest_expansion:
            retained = []
            for issue in report["blocking_issues_remaining"]:
                if issue.get("area") == "source_grounding" and any(marker in issue.get("message", "").casefold() for marker in ("short source", "limited source", "brief source", "short segment", "limited evidence")):
                    report["warnings"].append({"area": "source_grounding", "message": f"Warning only: {issue['message']}"})
                else:
                    retained.append(issue)
            report["blocking_issues_remaining"] = retained
    for key in ("warnings", "blocking_issues_fixed", "blocking_issues_remaining"):
        report[key] = _dedupe_issues(report[key])
    return report
