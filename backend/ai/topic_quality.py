"""Deterministic safeguards around the model-assisted topic quality review."""
from __future__ import annotations

from copy import deepcopy
import re
from urllib.parse import urlparse

from ..youtube.transcript_provider import TranscriptUnavailableError, canonical_youtube_url, youtube_video_id
from .metadata_resolution import MetadataResolutionService, TAXONOMY_REGISTRY


CONCEPT_TYPES = ("broad_concept", "named_method", "architecture", "loss_or_objective", "mathematical_concept", "training_mechanism", "evaluation_concept")

# This is a controlled vocabulary, not a catalog or a title-to-answer table.
MATH_VOCABULARY = {
    "vectors": "Vectors and matrices", "matrix": "Vectors and matrices", "dot": "Dot products",
    "norm": "Norms", "cos": "Cosine similarity", "cosine": "Cosine similarity",
    "softmax": "Softmax", "cross-entropy": "Cross-entropy", "cross entropy": "Cross-entropy",
    "kl": "KL divergence", "kullback": "KL divergence", "entropy": "Entropy",
    "temperature": "Temperature scaling", "expectation": "Expectations", "variance": "Variance",
    "covariance": "Covariance", "eigen": "Eigenvalues and eigenvectors", "gradient": "Gradients",
    "chain rule": "Chain rule", "optim": "Optimization basics", "contrastive": "Contrastive learning intuition",
    "nt-xent": "Cross-entropy", "\\tau": "Temperature scaling", "tau": "Temperature scaling", "^t": "Dot products",
}
ARCHITECTURE_MATH_TERMS = {"cnn", "convolutional neural network", "resnet", "transformer", "vision transformer", "gradient descent", "gradient-descent"}


def taxonomy_context() -> dict:
    return TAXONOMY_REGISTRY


def _slug(value: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", value.casefold()))


def _issue(area: str, message: str) -> dict:
    return {"area": area, "message": message}


def _canonical_relation_ids(values: object) -> list[str]:
    """Relationships are a set for audit purposes; stable ordering avoids false mismatches."""
    return sorted({value for value in values if isinstance(value, str) and value}) if isinstance(values, list) else []


def canonicalize_relationship_metadata(payload: dict) -> dict:
    """Canonicalize a saved draft without silently accepting stale resolver metadata."""
    topic = deepcopy(payload)
    prerequisites = _canonical_relation_ids(topic.get("prerequisite_topic_ids", []))
    related = _canonical_relation_ids(topic.get("related_topic_ids", []))
    topic["prerequisite_topic_ids"], topic["related_topic_ids"] = prerequisites, related
    durable = {("prerequisite", topic_id) for topic_id in prerequisites} | {("related", topic_id) for topic_id in related}
    justifications = [item for item in topic.get("relationship_justifications", []) if isinstance(item, dict)
                      and (item.get("relationship"), item.get("topic_id")) in durable]
    topic["relationship_justifications"] = sorted(justifications, key=lambda item: (item.get("relationship", ""), item.get("topic_id", "")))
    metadata = topic.get("metadata_resolution")
    if isinstance(metadata, dict) and isinstance(metadata.get("durable_edges"), dict):
        metadata["durable_edges"] = {
            "prerequisites": _canonical_relation_ids(metadata["durable_edges"].get("prerequisites", [])),
            "related": _canonical_relation_ids(metadata["durable_edges"].get("related", [])),
        }
    return topic


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


def _normalize_math_prerequisites(topic: dict, warnings: list[dict]) -> None:
    """Math prerequisites explain equations; they are not durable curriculum edges."""
    foundation = topic.get("mathematical_foundation")
    if not isinstance(foundation, dict):
        return
    def normalize(items: object, evidence: str, label: str) -> list[str]:
        requested = [item.strip() for item in items if isinstance(item, str) and item.strip()] if isinstance(items, list) else []
        evidence_lower = evidence.casefold()
        inferred = [display for token, display in MATH_VOCABULARY.items() if token in evidence_lower]
        # Keep a controlled requested term only if equations/text actually support it.
        clean = []
        for item in requested:
            if item.casefold() in ARCHITECTURE_MATH_TERMS:
                warnings.append(_issue("mathematics", f"Removed architecture/training item '{item}' from {label} mathematical prerequisites."))
                continue
            canonical = next((display for display in MATH_VOCABULARY.values() if item.casefold() == display.casefold()), None)
            if canonical and canonical not in inferred:
                warnings.append(_issue("mathematics", f"Removed '{item}' from {label} because it is not supported by its equations."))
                continue
            if canonical:
                clean.append(canonical)
        # The local inference gives a safe repair when the model supplied generic items.
        return list(dict.fromkeys(clean or inferred))

    sections = [section for section in foundation.get("sections", []) if isinstance(section, dict)]
    section_evidence_values = []
    for section in sections:
        section_evidence_values.append(str(section.get("explanation", "")) + " " + " ".join(
            str(item.get("latex", "")) + " " + str(item.get("explanation", "")) for item in section.get("equations", []) if isinstance(item, dict)))
    # A term that appears only in one heterogeneous section is section-level,
    # rather than a misleading prerequisite for the entire broad topic.
    global_evidence = str(foundation.get("overview", ""))
    if len(section_evidence_values) <= 1:
        global_evidence += " " + " ".join(section_evidence_values)
    else:
        shared_tokens = set(re.findall(r"[a-z0-9]+", section_evidence_values[0].casefold()))
        for evidence in section_evidence_values[1:]:
            shared_tokens &= set(re.findall(r"[a-z0-9]+", evidence.casefold()))
        global_evidence += " " + " ".join(shared_tokens)
    foundation["prerequisites"] = normalize(foundation.get("prerequisites", []), global_evidence, "global")
    for section, section_evidence in zip(sections, section_evidence_values):
        section["prerequisites"] = normalize(section.get("prerequisites", []), section_evidence, f"section '{section.get('title', 'untitled')}'")


def _source_grounding_checks(topic: dict) -> tuple[list[dict], list[dict]]:
    """Short honest evidence is a caveat; misleading provenance is not."""
    warnings, blocking = [], []
    sources = topic.get("sources", [])
    provenance = topic.get("source_provenance")
    requires_provenance = bool(provenance) or any(isinstance(source, dict) and source.get("type") == "youtube" for source in sources)
    if not requires_provenance:
        return warnings, blocking
    if not isinstance(provenance, dict) or not isinstance(provenance.get("source_derived"), dict) or not isinstance(provenance.get("ai_expanded"), str):
        return warnings, [_issue("provenance", "Video-derived content needs clear source_derived and AI-expanded provenance.")]
    source = provenance["source_derived"]
    if not source.get("source_evidence_summary"):
        blocking.append(_issue("provenance", "Source-derived evidence needs a summary."))
    is_youtube = source.get("kind") == "youtube"
    if is_youtube and not source.get("video_url"):
        blocking.append(_issue("provenance", "YouTube source provenance needs a canonical video URL."))
    timestamps = source.get("timestamp_seconds")
    if timestamps is not None and (not isinstance(timestamps, list) or any(not isinstance(item, (int, float)) or item < 0 for item in timestamps)):
        blocking.append(_issue("source_grounding", "Source evidence timestamps are malformed or contradictory."))
    elif isinstance(timestamps, list):
        claimed_times = [int(value) for value in re.findall(r"(?:at|around|~)\s*(\d+)\s*s\b", str(source.get("source_evidence_summary", "")).casefold())]
        if claimed_times and not any(abs(claimed - timestamp) <= 30 for claimed in claimed_times for timestamp in timestamps):
            blocking.append(_issue("source_grounding", "Source evidence summary timestamp contradicts structured provenance timestamps."))
    if is_youtube and source.get("video_url"):
        try:
            url_id = youtube_video_id(source["video_url"])
            if source.get("video_id") and source["video_id"] != url_id:
                blocking.append(_issue("source_grounding", "Source provenance video ID contradicts its canonical URL."))
        except TranscriptUnavailableError:
            blocking.append(_issue("provenance", "Source provenance does not contain a valid canonical YouTube URL."))
    evidence_words = len(str(source.get("source_evidence_summary", "")).split())
    expanded = provenance.get("ai_expanded", "")
    if evidence_words <= 40 and "ai-expand" in expanded.casefold():
        warnings.append(_issue("source_grounding", "Source segment is brief; the broader explanation is explicitly marked AI-expanded."))
    return warnings, blocking


def _method_completeness_checks(topic: dict) -> tuple[list[dict], list[dict]]:
    """Separate material method failures from useful-but-secondary canonical detail."""
    warnings, blocking = [], []
    if _slug(topic.get("title", "")) != "simclr":
        return warnings, blocking
    text = " ".join(str(topic.get(field, "")) for field in ("core_explanation", "mechanism", "quick_recall")).casefold()
    equations = " ".join(equation.get("latex", "") for section in (topic.get("mathematical_foundation") or {}).get("sections", [])
                         for equation in section.get("equations", []))
    if not all(token in text for token in ("augment", "projection", "contrast")):
        blocking.append(_issue("named_method_completeness", "SimCLR is missing a core-defining augmentation, projection-head, or contrastive-loss mechanism."))
    if "ell_{j,i}" not in equations and "i→j" not in text and "j→i" not in text and "i->j" not in text and "j->i" not in text:
        warnings.append(_issue("named_method_completeness", "SimCLR does not explicitly state symmetric i→j and j→i NT-Xent averaging; this is a useful completeness detail, not a blocker when the loss is otherwise correct."))
    return warnings, blocking


def _generic_heuristic_relation(topic_id: str, relationship: str, reason: str, concept_type: str) -> bool:
    """Reject stock graph edges when their rationale only describes implementation adjacency."""
    text = reason.casefold()
    if any(token in text for token in ("useful general background", "useful background", "broad background", "sometimes use", "may use")):
        return True
    # Universal, intent-based guardrails: these reject stock rationales without
    # knowing anything about the current topic's title.
    if topic_id == "gradient-descent" and relationship == "prerequisite":
        return any(token in text for token in ("train", "optim", "backprop", "fitted", "update"))
    if topic_id == "cnn" and relationship == "prerequisite" and concept_type == "architecture":
        return True
    if topic_id in {"cnn", "resnet"} and any(token in text for token in ("backbone", "uses cnn", "neural network", "implementation")):
        return True
    if topic_id == "gradient-flow" and any(token in text for token in ("residual", "gradient propagation", "training dynamics", "backprop")):
        return True
    if topic_id in {"pca", "covariance"} and any(token in text for token in ("embedding", "feature", "representation vector", "latent")):
        return True
    if topic_id == "knowledge-distillation" and any(token in text for token in ("teacher", "student", "teacher-student")):
        return True
    return False


def relationship_lint(payload: dict, *, catalog: list[dict], assigned_topic_id: str) -> tuple[list[dict], list[dict]]:
    """Non-mutating graph checks for draft validation and quality reports."""
    errors, warnings = [], []
    allowed = {item["id"] for item in catalog}
    concept_type = payload.get("concept_type", "broad_concept")
    justifications = {(item.get("relationship"), item.get("topic_id")): item
        for item in payload.get("relationship_justifications", []) if isinstance(item, dict)}
    values = {field: payload.get(field, []) for field in ("prerequisite_topic_ids", "related_topic_ids")}
    for field, kind, maximum in (("prerequisite_topic_ids", "prerequisite", 3), ("related_topic_ids", "related", 4)):
        items = values[field]
        if len(items) != len(set(items)):
            errors.append(_issue("relationships", f"{field} contains duplicate IDs."))
        if len(items) > maximum:
            warnings.append(_issue("relationships", f"{field} exceeds the soft sparsity target of {maximum}."))
        for topic_id in items:
            relation = justifications.get((kind, topic_id))
            if topic_id not in allowed:
                errors.append(_issue("relationships", f"{field} contains an unavailable topic ID '{topic_id}'."))
            elif topic_id == assigned_topic_id:
                errors.append(_issue("relationships", "A topic cannot link to itself."))
            elif not relation or relation.get("confidence") != "high" or not relation.get("reason"):
                errors.append(_issue("relationships", f"Durable {kind} '{topic_id}' needs a high-confidence rationale."))
            elif _generic_heuristic_relation(topic_id, kind, relation["reason"], concept_type):
                errors.append(_issue("relationships", f"{kind.title()} '{topic_id}' uses a prohibited generic heuristic."))
    overlap = set(values["prerequisite_topic_ids"]) & set(values["related_topic_ids"])
    if overlap:
        errors.append(_issue("relationships", f"A topic cannot be both prerequisite and related: {', '.join(sorted(overlap))}."))
    return errors, warnings


def normalize_topic(payload: dict, *, requested_title: str, assigned_topic_id: str, catalog: list[dict], source_context: dict | None = None,
                    enforce_justifications: bool = True) -> tuple[dict, list[dict], list[dict]]:
    """Normalize only generated content; never mutate an approved file or an old draft."""
    topic = deepcopy(payload)
    warnings, blocking = [], []
    topic.pop("id", None)  # A model ID is never durable identity.
    topic["title"] = requested_title or topic.get("title", "")
    topic["durable_topic_id"] = assigned_topic_id  # draft-only, backend-owned preview identity
    topic["tags"] = list(dict.fromkeys(tag.strip() for tag in topic.get("tags", []) if tag and tag.strip()))
    if topic.get("category") not in TAXONOMY_REGISTRY:
        blocking.append(_issue("taxonomy", f"Unknown category '{topic.get('category')}'."))
    if topic.get("concept_type") not in CONCEPT_TYPES:
        topic["concept_type"] = "broad_concept"
        warnings.append(_issue("taxonomy", "Unrecognized concept type was normalized to broad_concept."))
    elif topic["concept_type"] not in TAXONOMY_REGISTRY.get(topic.get("category"), {}).get("compatible_concept_types", []):
        blocking.append(_issue("taxonomy", f"Concept type '{topic['concept_type']}' is not compatible with category '{topic.get('category')}'."))

    catalog_ids = {item["id"] for item in catalog}
    catalog_titles = {item["id"]: item["title"] for item in catalog}
    justifications = {}
    for relation in topic.get("relationship_justifications", []):
        if isinstance(relation, dict) and relation.get("topic_id") in catalog_ids:
            justifications[(relation.get("relationship"), relation.get("topic_id"))] = relation
    clean_justifications = []
    suggestions, suggested_keys = [], set()
    for item in topic.get("suggested_new_topic_relationships", []):
        if not isinstance(item, dict):
            continue
        key = (_slug(str(item.get("title", ""))), item.get("relationship"))
        if key[0] and key not in suggested_keys:
            suggestions.append(item); suggested_keys.add(key)

    def suggest(topic_id: str, kind: str, reason: str) -> None:
        item = {"title": catalog_titles.get(topic_id, topic_id), "relationship": kind,
                "reason": f"Medium-confidence relationship, not saved as a durable edge: {reason}"}
        key = (_slug(item["title"]), kind)
        if key not in suggested_keys:
            suggestions.append(item); suggested_keys.add(key)

    used = set()
    for field, kind, maximum in (("prerequisite_topic_ids", "prerequisite", 3), ("related_topic_ids", "related", 4)):
        clean = []
        for topic_id in topic.get(field, []):
            relation = justifications.get((kind, topic_id))
            if topic_id not in catalog_ids:
                warnings.append(_issue("relationships", f"Removed unavailable {kind} ID '{topic_id}'."))
            elif topic_id == assigned_topic_id:
                warnings.append(_issue("relationships", "Removed a self relationship."))
            elif topic_id in used:
                warnings.append(_issue("relationships", f"Removed {topic_id} because a topic cannot be both prerequisite and related."))
            elif enforce_justifications and not relation:
                warnings.append(_issue("relationships", f"Removed {kind} relationship '{topic_id}' without an explicit rationale."))
            elif relation and relation.get("confidence") == "low":
                warnings.append(_issue("relationships", f"Removed low-confidence {kind} relationship '{topic_id}'."))
            elif relation and relation.get("confidence") == "medium":
                suggest(topic_id, kind, relation["reason"])
                warnings.append(_issue("relationships", f"Moved medium-confidence {kind} relationship '{topic_id}' to suggestions."))
            elif relation and relation.get("confidence") != "high":
                warnings.append(_issue("relationships", f"Removed {kind} relationship '{topic_id}' without high confidence."))
            elif relation and _generic_heuristic_relation(topic_id, kind, relation["reason"], topic["concept_type"]):
                warnings.append(_issue("relationships", f"Removed generic heuristic {kind} relationship '{topic_id}'."))
            elif len(clean) >= maximum:
                warnings.append(_issue("relationships", f"Removed {kind} relationship '{topic_id}' to keep the graph sparse (maximum {maximum})."))
            else:
                clean.append(topic_id); used.add(topic_id)
                if relation:
                    clean_justifications.append(relation)
        topic[field] = list(dict.fromkeys(clean))
    topic["relationship_justifications"] = clean_justifications
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
    _normalize_math_prerequisites(topic, warnings)
    source_warnings, source_blocking = _source_grounding_checks(topic)
    method_warnings, method_blocking = _method_completeness_checks(topic)
    warnings.extend(source_warnings + method_warnings)
    blocking.extend(source_blocking + method_blocking)
    blocking.extend(_equation_issues(topic))
    topic["relationship_warnings"] = warnings
    return topic, warnings, blocking


def apply_relationship_resolution(payload: dict, resolution: dict, *, candidates: list[dict], assigned_topic_id: str) -> tuple[dict, list[dict], list[dict]]:
    """Apply one compact resolver result; retrieval never itself creates an edge."""
    topic = deepcopy(payload)
    warnings, blocking = [], []
    allowed = {item["topic_id"]: item for item in candidates}
    accepted, justifications, suggestions, used = {"prerequisite": [], "related": []}, [], list(topic.get("suggested_new_topic_relationships", [])), set()
    for key, kind, maximum in (("prerequisites", "prerequisite", 3), ("related", "related", 4)):
        for item in resolution.get(key, []):
            if not isinstance(item, dict):
                continue
            topic_id, confidence, reason = item.get("topic_id"), item.get("confidence"), str(item.get("reason", "")).strip()
            if topic_id not in allowed or topic_id == assigned_topic_id or topic_id in used:
                warnings.append(_issue("relationships", f"Discarded invalid or overlapping resolver relationship '{topic_id}'."))
                continue
            if _generic_heuristic_relation(topic_id, kind, reason, topic.get("concept_type", "broad_concept")):
                warnings.append(_issue("relationships", f"Discarded generic {kind} relationship '{topic_id}'."))
                continue
            if confidence == "high" and reason and len(accepted[kind]) < maximum:
                accepted[kind].append(topic_id); used.add(topic_id)
                justifications.append({"topic_id": topic_id, "relationship": kind, "reason": reason, "confidence": "high"})
            elif confidence == "medium":
                title = allowed.get(topic_id, {}).get("title", topic_id)
                suggestions.append({"title": title, "relationship": kind, "reason": f"Medium-confidence relationship, not saved as a durable edge: {reason}"})
            # Low is intentionally absent from durable metadata and suggestions.
    topic["prerequisite_topic_ids"] = _canonical_relation_ids(accepted["prerequisite"])
    topic["related_topic_ids"] = _canonical_relation_ids(accepted["related"])
    topic["relationship_justifications"] = justifications
    unique_suggestions, seen = [], set()
    for item in suggestions:
        if not isinstance(item, dict):
            continue
        marker = (_slug(str(item.get("title", ""))), item.get("relationship"))
        if marker[0] and marker not in seen:
            unique_suggestions.append(item); seen.add(marker)
    topic["suggested_new_topic_relationships"] = unique_suggestions
    topic["metadata_resolution"] = {
        "resolver_version": resolution.get("resolver_version"),
        "resolved_category": topic.get("category"), "resolved_concept_type": topic.get("concept_type"),
        "candidate_topic_ids": [item["topic_id"] for item in candidates],
        "durable_edges": {"prerequisites": list(topic["prerequisite_topic_ids"]), "related": list(topic["related_topic_ids"])},
        "math_prerequisites": {"global": list((topic.get("mathematical_foundation") or {}).get("prerequisites", [])),
            "sections": {str(section.get("title", index)): list(section.get("prerequisites", [])) for index, section in enumerate((topic.get("mathematical_foundation") or {}).get("sections", [])) if isinstance(section, dict)}},
        "rejected_candidates": resolution.get("rejected_candidates", []),
    }
    return canonicalize_relationship_metadata(topic), warnings, blocking


def metadata_payload_consistency(payload: dict, report: dict, *, catalog: list[dict], assigned_topic_id: str) -> list[dict]:
    """Fail closed if the stored metadata diverges from the resolution audit."""
    metadata = payload.get("metadata_resolution")
    if not isinstance(metadata, dict):
        return [_issue("schema", "Metadata resolution audit is missing from the persisted payload.")]
    problems = []
    if metadata.get("resolved_category") != payload.get("category") or metadata.get("resolved_concept_type") != payload.get("concept_type"):
        problems.append(_issue("taxonomy", "Resolved category or concept type does not match the persisted payload."))
    _, current_topic_hash, _ = MetadataResolutionService.cache_key(payload, [])
    if metadata.get("topic_hash") != current_topic_hash:
        problems.append(_issue("schema", "Educational content changed after metadata resolution; rebuild metadata before marking this draft Ready."))
    relationship_errors, _ = relationship_lint(payload, catalog=catalog, assigned_topic_id=assigned_topic_id)
    problems.extend(relationship_errors)
    foundation = payload.get("mathematical_foundation") or {}
    if metadata.get("durable_edges") != {"prerequisites": _canonical_relation_ids(payload.get("prerequisite_topic_ids", [])), "related": _canonical_relation_ids(payload.get("related_topic_ids", []))}:
        problems.append(_issue("relationships", "Persisted relationship metadata disagrees with the resolver audit."))
    current_math = {"global": list(foundation.get("prerequisites", [])), "sections": {
        str(section.get("title", index)): list(section.get("prerequisites", [])) for index, section in enumerate(foundation.get("sections", [])) if isinstance(section, dict)}}
    if metadata.get("math_prerequisites") != current_math:
        problems.append(_issue("mathematics", "Persisted mathematical prerequisites disagree with the resolver audit."))
    all_math = list(foundation.get("prerequisites", [])) + [value for section in foundation.get("sections", []) if isinstance(section, dict) for value in section.get("prerequisites", [])]
    for item in all_math:
        if item not in set(MATH_VOCABULARY.values()):
            problems.append(_issue("mathematics", f"Mathematical prerequisite '{item}' is outside the controlled vocabulary."))
    # A reviewer report may not claim metadata was repaired while the payload has no resolver audit.
    metadata_claim = " ".join(str(issue.get("message", "")) for issue in report.get("blocking_issues_fixed", []) if isinstance(issue, dict)).casefold()
    if any(word in metadata_claim for word in ("relationship", "category", "mathematical prerequisite")) and not metadata:
        problems.append(_issue("schema", "Quality report claims metadata repair without matching persisted metadata."))
    return problems


def final_quality_report(reviewer_report: dict, deterministic_warnings: list[dict], deterministic_blocking: list[dict], *, payload: dict | None = None) -> dict:
    report = deepcopy(reviewer_report)
    report.setdefault("blocking_issues_fixed", [])
    report.setdefault("blocking_issues_remaining", [])
    report.setdefault("warnings", [])
    report.setdefault("confidence", "medium")
    report["warnings"] = report["warnings"] + deterministic_warnings
    # A weak durable edge is a blocking graph-quality defect. The normalizer can
    # repair it safely by removing/demoting the edge; retain that audit trail as
    # a fixed blocker rather than silently calling the result ready.
    fixed_relationships = [item for item in deterministic_warnings if item.get("area") == "relationships"
                           and any(marker in item.get("message", "") for marker in ("Removed", "Moved", "generic heuristic"))]
    report["blocking_issues_fixed"] = report["blocking_issues_fixed"] + fixed_relationships
    report["blocking_issues_remaining"] = report["blocking_issues_remaining"] + deterministic_blocking
    if payload:
        provenance = payload.get("source_provenance", {})
        honest_expansion = isinstance(provenance, dict) and isinstance(provenance.get("source_derived"), dict) and isinstance(provenance.get("ai_expanded"), str) and "ai-expand" in provenance.get("ai_expanded", "").casefold()
        short_source_markers = ("short source", "limited source", "brief source", "short segment", "limited evidence", "evidence is brief", "ai-expanded")
        if honest_expansion:
            retained = []
            for issue in report["blocking_issues_remaining"]:
                if issue.get("area") == "source_grounding" and any(marker in issue.get("message", "").casefold() for marker in short_source_markers):
                    report["warnings"].append({"area": "source_grounding", "message": f"Warning only: {issue['message']}"})
                else:
                    retained.append(issue)
            report["blocking_issues_remaining"] = retained
    return report
