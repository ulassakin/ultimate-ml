"""Quality normalisation keeps educational checks and drops retired graph output."""
from backend.ai.topic_quality import normalize_topic


def test_normalization_drops_graph_metadata_but_keeps_provenance_and_math_checks():
    payload = {
        "title": "CLIP", "category": "representation_learning", "difficulty": "intermediate", "concept_type": "named_method",
        "tags": ["multimodal"], "one_sentence_summary": "Joint image and text embeddings.", "quick_recall": "Match image and text representations.",
        "core_explanation": "CLIP learns aligned image and text representations.",
        "prerequisite_topic_ids": ["covariance"], "related_topic_ids": ["cnn"], "metadata_resolution": {"legacy": True},
        "mathematical_foundation": {"overview": "Similarity scores.", "prerequisites": ["CNN", "Dot products"], "sections": [{"title": "Scores", "explanation": "Compare embeddings.", "equations": [{"latex": "s=x^Ty", "explanation": "A dot product score."}]}]},
    }
    normalized, warnings, blocking = normalize_topic(payload, requested_title="CLIP", assigned_topic_id="clip")
    assert not blocking
    assert normalized["mathematical_foundation"]["prerequisites"] == ["Dot products"]
    assert not any(field in normalized for field in ("prerequisite_topic_ids", "related_topic_ids", "metadata_resolution"))
    assert warnings

