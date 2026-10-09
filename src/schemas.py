from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Topic(str, Enum):
    ARRAYS              = "Arrays"
    LINKED_LISTS        = "Linked Lists"
    TREES               = "Trees"
    DYNAMIC_PROGRAMMING = "Dynamic Programming"
    GRAPHS              = "Graphs"
    OOPS                = "OOPs"
    OPERATING_SYSTEMS   = "Operating Systems"
    SYSTEM_DESIGN       = "System Design"
    PYTHON              = "Python"
    MACHINE_LEARNING    = "Machine Learning"
    HR                  = "HR"


Difficulty = Literal["easy", "medium", "hard"]


class GeneratedQuestion(BaseModel):
    question: str
    topic: Topic
    difficulty: Difficulty
    expected_points: List[str]
    hint: Optional[str] = None


class CriterionScore(BaseModel):
    criterion: str
    met: bool
    comment: str


class EvaluationResult(BaseModel):
    overall_score: int = Field(ge=0, le=10)
    criterion_scores: List[CriterionScore]
    strengths: str
    improvements: str
    ideal_answer_summary: str
