from typing import Literal

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    text: str = Field(min_length=1)
    start_seconds: float | None = None
    duration_seconds: float | None = None


class TranscriptSource(BaseModel):
    kind: Literal["youtube", "pasted"]
    video_url: str | None = None
    video_id: str | None = None
    title: str = "Untitled learning source"
    channel: str | None = None
    language: str | None = None
    duration_seconds: float | None = None
    transcript_attribution: str


class TranscriptDocument(BaseModel):
    source: TranscriptSource
    segments: list[TranscriptSegment] = Field(min_length=1)


class VideoConcept(BaseModel):
    canonical_name: str = Field(min_length=1)
    importance: Literal["core", "supporting", "incidental"]
    concept_type: Literal["broad_concept", "named_method", "architecture", "mathematical_concept", "training_mechanism", "evaluation_concept", "supporting_idea"] = "broad_concept"
    parent_concepts: list[str] = []
    source_evidence_summary: str = Field(min_length=1)
    ml_learning_value: str = Field(min_length=1)
    timestamp_seconds: list[float] = []


class VideoConceptBatch(BaseModel):
    concepts: list[VideoConcept] = Field(min_length=1, max_length=16)
    learning_arc: str = Field(min_length=1)
