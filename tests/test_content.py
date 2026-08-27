import json
from pathlib import Path
import pytest
from backend.content import ContentError, load_library

def test_seed_content_is_valid_and_connected():
    library = load_library()
    assert "dino" in library.topics
    assert library.questions["dinov3_q001"]["concept_refresher_topic_id"] == "dinov3"


def test_v2_math_content_and_v1_content_load_together():
    library = load_library()
    assert library.topics["knowledge-distillation"]["content_version"] == 1
    vectors = library.topics["vectors-and-vector-spaces"]
    assert vectors["prerequisite_topic_ids"] == []
    equation = vectors["mathematical_foundation"]["sections"][0]["equations"][0]
    assert equation["latex"] and equation["explanation"]
    assert library.questions["dinov3_q001"]["direct_answer"]

def test_legacy_relationship_metadata_is_tolerated(tmp_path: Path):
    (tmp_path / "topics").mkdir(); (tmp_path / "questions").mkdir()
    topic = {"id":"one","title":"One","category":"fundamentals","difficulty":"beginner","quick_recall":"q","core_explanation":"c","related_topics":["missing"]}
    (tmp_path / "topics" / "one.json").write_text(json.dumps(topic))
    assert load_library(tmp_path).topics["one"]["related_topics"] == ["missing"]


def test_malformed_equation_is_reported(tmp_path: Path):
    (tmp_path / "topics").mkdir(); (tmp_path / "questions").mkdir()
    topic = {"id":"one","title":"One","category":"mathematical_foundations","difficulty":"beginner","quick_recall":"q","core_explanation":"c", "mathematical_foundation":{"sections":[{"title":"Bad","explanation":"x","equations":[{"latex":"x"}]}]}}
    (tmp_path / "topics" / "one.json").write_text(json.dumps(topic))
    with pytest.raises(ContentError, match="equations need latex and explanation"):
        load_library(tmp_path)
