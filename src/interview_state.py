from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

from src.schemas import Topic, EvaluationResult

TECHNICAL_Q_LIMIT = 5
HR_Q_LIMIT = 3


class InterviewStage(IntEnum):
    SETUP     = 0
    INTRO     = 1
    TECHNICAL = 2
    HR        = 3
    REPORT    = 4


@dataclass
class QuestionRecord:
    question_number: int
    question_text: str
    topic: Topic
    difficulty: str
    user_answer: str
    overall_score: int
    strengths: str
    improvements: str
    ideal_answer_summary: str
    flow: str


@dataclass
class InterviewState:
    stage: InterviewStage = InterviewStage.SETUP
    resume_file_hash: Optional[str] = None
    resume_text: str = ""
    jd_text: str = ""

    message_history: List[Dict[str, str]] = field(default_factory=list)

    current_kb_entry_id: Optional[str] = None
    current_kb_question: str = ""
    current_question_text: str = ""
    current_rubric: List[str] = field(default_factory=list)
    current_reference: Optional[str] = None
    current_topic: Optional[Topic] = None
    current_difficulty: str = "medium"
    current_flow: str = ""

    awaiting_answer: bool = False
    last_evaluation: Optional[Any] = None

    topics_asked: List[Topic] = field(default_factory=list)
    question_records: List[QuestionRecord] = field(default_factory=list)
    technical_q_count: int = 0
    hr_q_count: int = 0

    def average_score(self) -> float:
        if not self.question_records:
            return 0.0
        return sum(r.overall_score for r in self.question_records) / len(self.question_records)

    def scores_by_topic(self) -> Dict[str, float]:
        accumulator: Dict[str, List[int]] = {}
        for record in self.question_records:
            key = record.topic.value
            accumulator.setdefault(key, []).append(record.overall_score)
        return {k: sum(v) / len(v) for k, v in accumulator.items()}

    def weak_topics(self) -> List[Topic]:
        return [
            Topic(k)
            for k, v in self.scores_by_topic().items()
            if v < 5
        ]

    def topics_asked_values(self) -> List[str]:
        return [t.value for t in self.topics_asked]

    def weak_topics_values(self) -> List[str]:
        return [t.value for t in self.weak_topics()]

    def suggested_difficulty(self) -> str:
        avg = self.average_score()
        if avg > 7:
            return "hard"
        if avg < 5:
            return "easy"
        return "medium"

    def is_technical_done(self) -> bool:
        return self.technical_q_count >= TECHNICAL_Q_LIMIT

    def is_hr_done(self) -> bool:
        return self.hr_q_count >= HR_Q_LIMIT

    def total_questions_done(self) -> int:
        return self.technical_q_count + self.hr_q_count

    def add_message(self, role: str, content: str) -> None:
        self.message_history.append({"role": role, "content": content})

    def advance_stage(self) -> None:
        self.stage = InterviewStage(self.stage + 1)

    def record_answer(
        self,
        user_answer: str,
        result: EvaluationResult,
    ) -> None:
        record = QuestionRecord(
            question_number=self.total_questions_done() + 1,
            question_text=self.current_question_text,
            topic=self.current_topic,
            difficulty=self.current_difficulty,
            user_answer=user_answer,
            overall_score=result.overall_score,
            strengths=result.strengths,
            improvements=result.improvements,
            ideal_answer_summary=result.ideal_answer_summary,
            flow=self.current_flow,
        )
        self.question_records.append(record)
        if self.current_topic and self.current_topic not in self.topics_asked:
            self.topics_asked.append(self.current_topic)

        if self.stage == InterviewStage.TECHNICAL:
            self.technical_q_count += 1
        elif self.stage == InterviewStage.HR:
            self.hr_q_count += 1

    def clear_current_question(self) -> None:
        self.current_kb_entry_id = None
        self.current_kb_question = ""
        self.current_question_text = ""
        self.current_rubric = []
        self.current_reference = None
        self.current_topic = None
        self.current_difficulty = "medium"
        self.current_flow = ""
        self.awaiting_answer = False
        self.last_evaluation = None
