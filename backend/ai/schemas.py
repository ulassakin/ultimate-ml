from typing import Literal
from pydantic import BaseModel, Field
from ..domain import Difficulty


class EquationDraft(BaseModel):
    latex: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class MathSectionDraft(BaseModel):
    title: str
    explanation: str
    equations: list[EquationDraft] = []
    prerequisites: list[str] = []


class MathematicalFoundationDraft(BaseModel):
    overview: str = ""
    prerequisites: list[str] = []
    sections: list[MathSectionDraft] = []


class TopicDraft(BaseModel):
    title: str
    category: str
    difficulty: Difficulty
    concept_type: Literal["broad_concept", "named_method", "architecture", "loss_or_objective", "mathematical_concept", "training_mechanism", "evaluation_concept"] = "broad_concept"
    tags: list[str] = []
    one_sentence_summary: str
    quick_recall: str
    big_picture: str = ""
    why_it_exists: str = ""
    intuition: str = ""
    core_explanation: str
    mechanism: str = ""
    ml_relevance: str = ""
    practical_example: str = ""
    mathematical_foundation: MathematicalFoundationDraft | None = None
    common_misconceptions: list[str] = []
    limitations: list[str] = []
    mental_models: list[str] = []
    deep_dive: str = ""


class QualityIssue(BaseModel):
    area: Literal["technical_correctness", "taxonomy", "difficulty", "tags", "mathematics", "named_method_completeness", "source_grounding", "provenance", "overclaims", "schema"]
    message: str = Field(min_length=1)


class TopicQualityReport(BaseModel):
    blocking_issues_fixed: list[QualityIssue] = []
    blocking_issues_remaining: list[QualityIssue] = []
    warnings: list[QualityIssue] = []
    confidence: Literal["high", "medium", "low"] = "medium"


class TopicQualityChange(BaseModel):
    """A concise claim about a material edit; verified against a backend diff."""
    field_path: str = Field(min_length=1)
    change_type: Literal["replace", "remove", "add"]
    reason: str = Field(min_length=1)


class TopicQualityReview(BaseModel):
    corrected_topic: TopicDraft
    quality_report: TopicQualityReport
    changes: list[TopicQualityChange] = []


class QuestionDraftItem(BaseModel):
    question_category: Literal["foundation", "intuition", "mechanism", "architecture", "mathematical_intuition", "comparison", "misconception", "application", "research_relevance"]
    difficulty: Difficulty
    question: str
    direct_answer: str
    expanded_answer: str
    key_points: list[str] = []
    common_wrong_ideas: list[str] = []


class QuestionDraftBatch(BaseModel):
    questions: list[QuestionDraftItem] = Field(min_length=1, max_length=12)


class RegeneratedSection(BaseModel):
    value: str | MathematicalFoundationDraft | list[str]
