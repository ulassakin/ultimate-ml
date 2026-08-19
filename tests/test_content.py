import json
from pathlib import Path
import pytest
from backend.content import ContentError, load_library

def test_seed_content_is_valid_and_connected():
    library = load_library()
    assert "resnet" in library.topics
    assert library.questions["resnet_q001"]["concept_refresher_topic_id"] == "resnet"

def test_missing_reference_is_reported(tmp_path: Path):
    (tmp_path / "topics").mkdir(); (tmp_path / "questions").mkdir()
    topic = {"id":"one","title":"One","category":"fundamentals","difficulty":"beginner","quick_recall":"q","core_explanation":"c","related_topics":["missing"]}
    (tmp_path / "topics" / "one.json").write_text(json.dumps(topic))
    with pytest.raises(ContentError, match="missing related topic 'missing'"):
        load_library(tmp_path)

