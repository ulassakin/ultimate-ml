from typing import Literal
from pydantic import BaseModel, Field


class EquationDraft(BaseModel):
    latex: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class MathSectionDraft(BaseModel):
    title: str
    explanation: str
    equations: list[EquationDraft] = []


class MathematicalFoundationDraft(BaseModel):
    overview: str = ""
    prerequisites: list[str] = []
    sections: list[MathSectionDraft] = []


class TopicDraft(BaseModel):
    title: str
    category: str
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
    deep_dive: str = ""


class QuestionDraftItem(BaseModel):
    question_category: Literal["foundation", "intuition", "mechanism", "architecture", "mathematical_intuition", "comparison", "misconception", "application", "research_relevance"]
    difficulty: Literal["beginner", "intermediate", "advanced"]
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
