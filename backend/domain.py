from enum import Enum


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ExplanationDepth(str, Enum):
    STANDARD = "standard"
    DEEP = "deep"
    ULTIMATE = "ultimate"

