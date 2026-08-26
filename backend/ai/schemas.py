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


class SuggestedRelationship(BaseModel):
    title: str
    relationship: Literal["prerequisite", "related"]
    reason: str


class ExistingTopicRelationship(BaseModel):
    """An explicit reason is required before a catalog ID becomes a durable edge."""
    topic_id: str
    relationship: Literal["prerequisite", "related"]
    reason: str = Field(min_length=12)
    confidence: Literal["high", "medium", "low"]


class ResolvedRelationship(BaseModel):
    """One resolver decision; its enclosing list gives the relationship kind."""
    topic_id: str
    reason: str = Field(min_length=12)
    confidence: Literal["high", "medium", "low"]


class RejectedRelationshipCandidate(BaseModel):
    topic_id: str
    reason: str = Field(min_length=3)


class RelationshipResolution(BaseModel):
    prerequisites: list[ResolvedRelationship] = []
    related: list[ResolvedRelationship] = []
    rejected_candidates: list[RejectedRelationshipCandidate] = []


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
    prerequisite_topic_ids: list[str] = []
    related_topic_ids: list[str] = []
    relationship_justifications: list[ExistingTopicRelationship] = []
    suggested_new_topic_relationships: list[SuggestedRelationship] = []
    deep_dive: str = ""


class QualityIssue(BaseModel):
    area: Literal["technical_correctness", "taxonomy", "difficulty", "tags", "relationships", "mathematics", "named_method_completeness", "source_grounding", "provenance", "overclaims", "schema"]
    message: str = Field(min_length=1)


class TopicQualityReport(BaseModel):
    blocking_issues_fixed: list[QualityIssue] = []
    blocking_issues_remaining: list[QualityIssue] = []
    warnings: list[QualityIssue] = []
    confidence: Literal["high", "medium", "low"] = "medium"


class TopicQualityReview(BaseModel):
    corrected_topic: TopicDraft
    quality_report: TopicQualityReport


class QuestionDraftItem(BaseModel):
    question_category: Literal["foundation", "intuition", "mechanism", "architecture", "mathematical_intuition", "comparison", "misconception", "application", "research_relevance"]
    difficulty: Difficulty
    question: str
    direct_answer: str
    expanded_answer: str
    key_points: list[str] = []
    common_wrong_ideas: list[str] = []
    related_topic_ids: list[str] = []


class QuestionDraftBatch(BaseModel):
    questions: list[QuestionDraftItem] = Field(min_length=1, max_length=12)


class RegeneratedSection(BaseModel):
    value: str | MathematicalFoundationDraft | list[str]
