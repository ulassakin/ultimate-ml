"""Local retrieval and sparse metadata resolution; similarity never creates an edge by itself."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter


RELATIONSHIP_RESOLVER_PROMPT_VERSION = "relationship-resolver-v1"
DEFAULT_TOP_K = 15


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


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def compact_document(topic: dict) -> dict:
    return {"topic_id": topic["id"], "title": topic["title"], "one_sentence_summary": topic.get("one_sentence_summary", ""),
            "concept_type": topic.get("concept_type", "broad_concept"), "category": topic.get("category", "ml_fundamentals"),
            "tags": list(topic.get("tags", []))[:12]}


def compact_topic_payload(topic: dict) -> dict:
    return {"title": topic.get("title", ""), "one_sentence_summary": topic.get("one_sentence_summary", ""),
            "concept_type": topic.get("concept_type", "broad_concept"), "category": topic.get("category", "ml_fundamentals"),
            "tags": list(topic.get("tags", []))[:12], "core_explanation": topic.get("core_explanation", "")[:4000]}


class MetadataResolutionService:
    """Dependency-free TF-IDF/cosine fallback appropriate for local-first catalogs."""
    def __init__(self, database, top_k: int | None = None):
        self.database = database
        try:
            configured = int(os.getenv("ULTIMATE_ML_METADATA_TOP_K", str(DEFAULT_TOP_K)))
        except ValueError:
            configured = DEFAULT_TOP_K
        self.top_k = max(1, min(top_k or configured, 20))

    def sync_documents(self, topics: dict[str, dict]) -> list[dict]:
        stored = {item["topic_id"]: item for item in self.database.get_retrieval_documents()}
        documents = []
        for topic in topics.values():
            document = compact_document(topic)
            digest = _hash(document)
            previous = stored.get(topic["id"])
            if previous and previous["content_hash"] == digest:
                documents.append({"content_hash": previous["content_hash"], "document": previous["document"]})
            else:
                self.database.upsert_retrieval_document(topic["id"], digest, document)
                documents.append({"content_hash": digest, "document": document})
        return documents

    def retrieve(self, current_topic: dict, topics: dict[str, dict], top_k: int | None = None) -> list[dict]:
        documents = self.sync_documents(topics)
        current_id = current_topic.get("id")
        current_title = current_topic.get("title", "").casefold()
        candidates = [item for item in documents if item["document"]["topic_id"] != current_id and item["document"]["title"].casefold() != current_title]
        if not candidates:
            return []
        query = _tokens(" ".join(str(value) for value in compact_topic_payload(current_topic).values() if not isinstance(value, list)) + " " + " ".join(current_topic.get("tags", [])))
        query_counts = Counter(query)
        document_terms = [Counter(_tokens(" ".join([item["document"]["title"], item["document"]["one_sentence_summary"], item["document"]["concept_type"], item["document"]["category"], " ".join(item["document"]["tags"])]))) for item in candidates]
        doc_frequency = Counter(term for terms in document_terms for term in terms)
        count = len(candidates)
        def score(terms: Counter) -> float:
            numerator = sum(query_counts[term] * terms[term] * (math.log((count + 1) / (doc_frequency[term] + 1)) + 1) ** 2 for term in query_counts if term in terms)
            q_norm = math.sqrt(sum(value * value for value in query_counts.values())) or 1.0
            d_norm = math.sqrt(sum(value * value for value in terms.values())) or 1.0
            return numerator / (q_norm * d_norm)
        ranked = sorted(zip(candidates, document_terms), key=lambda pair: (-score(pair[1]), pair[0]["document"]["topic_id"]))
        limit = max(1, min(top_k or self.top_k, 20))
        return [{**item["document"], "content_hash": item["content_hash"], "retrieval_score": round(score(terms), 6)} for item, terms in ranked[:limit]]

    @staticmethod
    def cache_key(topic: dict, candidates: list[dict], resolver_version: str = RELATIONSHIP_RESOLVER_PROMPT_VERSION) -> tuple[str, str, list[dict]]:
        topic_hash = _hash(compact_topic_payload(topic))
        candidate_hashes = [{"topic_id": item["topic_id"], "content_hash": item["content_hash"]} for item in candidates]
        return _hash({"topic_hash": topic_hash, "candidates": candidate_hashes, "resolver_version": resolver_version}), topic_hash, candidate_hashes
