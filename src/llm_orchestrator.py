import os
import random
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schemas import EvaluationResult, GeneratedQuestion, Topic

AVAILABLE_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]

def _get_llm(model: str = "gemini-3.7-flash"):
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.7,
        google_api_key=api_key,
    )

_str_parser = StrOutputParser()

VALID_TOPICS = ", ".join(t.value for t in Topic)

_INTRO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI mock interviewer. Greet the candidate warmly and professionally. "
        "Tell them: (1) they will face 5 technical questions followed by 3 HR questions, "
        "(2) questions are personalised to their resume and the job they applied for, "
        "(3) they get a score and specific feedback after every answer, "
        "(4) a full performance report with a radar chart is generated at the end. "
        "Keep it to 4-5 sentences. Be encouraging but professional."
    )),
    ("human", "Start the interview session."),
])

_FLOW_A_PERSONALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior technical interviewer.\n\n"
        "You have a standard KB question and the candidate's resume context. "
        "Personalise the standard question so the candidate is asked about their REAL experience.\n\n"
        "Standard KB Question: {kb_question}\n"
        "Topic: {topic} | Difficulty: {difficulty}\n\n"
        "Resume Context (retrieved):\n{resume_context}\n\n"
        "Topics already asked (do NOT repeat): {topics_asked}\n"
        "Candidate's weak areas: {weak_topics}\n"
        "Average score so far: {avg_score}/10\n\n"
        "Rules:\n"
        "- If the resume context is relevant, weave a specific project or skill into the question.\n"
        "- If the resume context is NOT relevant to this topic, return the standard question unchanged.\n"
        "- Keep the question to 2-3 sentences maximum.\n"
        "- Do NOT reveal the grading rubric or hint at the ideal answer.\n"
        "Return ONLY the question text. No preamble, no labels."
    )),
    ("human", "Generate the personalised question."),
])

_FLOW_B_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior technical interviewer conducting a job interview.\n\n"
        "Generate ONE interview question that is specific to the candidate's resume and the JD. "
        "Also write 3-5 expected_points that a strong answer should cover — "
        "these MUST match the question you wrote (they become the grading rubric).\n\n"
        "Resume Context:\n{resume_context}\n\n"
        "Job Description:\n{jd_text}\n\n"
        "Stage: {stage}\n"
        "Topics already asked (do NOT repeat): {topics_asked}\n"
        "Candidate's weak areas: {weak_topics}\n"
        "Average score so far: {avg_score}/10 "
        "(ask harder if avg > 7, easier if avg < 5)\n\n"
        "Allowed topic values (use exactly one): {valid_topics}\n"
        "Allowed difficulty values: easy, medium, hard\n\n"
        "IMPORTANT RULES:\n"
        "- Generate a genuinely personalised question based on candidate's real experience.\n"
        "- STRICT RELEVANCE: Only ask about technologies, languages, and concepts that match the candidate's Resume Context or Job Description.\n"
        "- NEVER ask Machine Learning or specialized domain questions unless the candidate explicitly lists Machine Learning / Data Science in their resume!"
    )),
    ("human", "Generate the next personalised interview question."),
])

_EVALUATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict, fair, and honest technical interviewer evaluating a candidate's answer.\n\n"
        "Original KB Question (core concept being tested): {kb_question}\n"
        "Actual Question Asked to Candidate: {question_asked}\n"
        "Candidate's Answer: {user_answer}\n\n"
        "Grading Rubric (evaluate the answer against EACH of these criteria):\n{rubric}\n\n"
        "Reference Answer (ideal answer — treat as gold standard if provided; "
        "if not provided, grade purely against the rubric):\n{reference}\n\n"
        "Instructions:\n"
        "- Score each rubric criterion as met (true/false) with an honest, concise comment.\n"
        "- Overall score (0 to 10): reflect how many criteria were met and technical accuracy.\n"
        "- DO NOT FORCE STRENGTHS: If the candidate answered incorrectly, gave an empty or superficial response, "
        "or stated that they do not know, DO NOT invent false praise or manufacture strengths. "
        "In strengths, simply write 'None identified for this question.' "
        "Only highlight genuine strengths when the candidate demonstrated real technical understanding.\n"
        "- improvements: Provide actionable, constructive feedback explaining what to study or how to answer.\n"
        "- ideal_answer_summary: 2-3 sentences explaining the correct approach so the candidate learns."
    )),
    ("human", "Evaluate the candidate's answer."),
])


def generate_intro() -> str:
    for model_name in AVAILABLE_MODELS:
        try:
            chain = _INTRO_PROMPT | _get_llm(model=model_name) | _str_parser
            return chain.invoke({})
        except Exception:
            continue
    return (
        "Welcome to your interview! I will conduct your technical and behavioral rounds today, "
        "tailored to your resume and target role. After each answer, you will receive immediate feedback "
        "and scoring, followed by a comprehensive final report with a radar chart."
    )


def personalize_kb_question(
    kb_question: str,
    topic: str,
    difficulty: str,
    resume_context: str,
    topics_asked: List[str],
    weak_topics: List[str],
    avg_score: float,
) -> str:
    for model_name in AVAILABLE_MODELS:
        try:
            chain = _FLOW_A_PERSONALIZE_PROMPT | _get_llm(model=model_name) | _str_parser
            return chain.invoke({
                "kb_question": kb_question,
                "topic": topic,
                "difficulty": difficulty,
                "resume_context": resume_context or "No resume context available.",
                "topics_asked": ", ".join(topics_asked) if topics_asked else "None yet",
                "weak_topics": ", ".join(weak_topics) if weak_topics else "None identified yet",
                "avg_score": round(avg_score, 1),
            })
        except Exception:
            continue
    return kb_question


def generate_resume_question(
    resume_context: str,
    jd_text: str,
    stage: str,
    topics_asked: List[str],
    weak_topics: List[str],
    avg_score: float,
) -> GeneratedQuestion:
    last_err = None
    for model_name in AVAILABLE_MODELS:
        try:
            question_llm = _get_llm(model=model_name).with_structured_output(GeneratedQuestion)
            chain = _FLOW_B_GENERATE_PROMPT | question_llm
            return chain.invoke({
                "resume_context": resume_context or "No resume context available.",
                "jd_text": jd_text or "No job description provided.",
                "stage": stage,
                "topics_asked": ", ".join(topics_asked) if topics_asked else "None yet",
                "weak_topics": ", ".join(weak_topics) if weak_topics else "None identified yet",
                "avg_score": round(avg_score, 1),
                "valid_topics": VALID_TOPICS,
            })
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Question generation failed.")


def is_admission_of_not_knowing(text: str) -> bool:
    t = text.strip().lower()
    phrases = [
        "dont know", "don't know", "dont knw", "idk", "not sure",
        "no idea", "no clue", "have no idea", "skip", "pass",
        "im sorry", "i am sorry", "sorry", "cannot answer", "can't answer",
        "no answer", "not aware", "haven't learned", "never used",
    ]
    if len(t.split()) <= 10 and any(p in t for p in phrases):
        return True
    if len(t.strip()) < 4:
        return True
    return False


def evaluate_answer(
    kb_question: str,
    question_asked: str,
    rubric: List[str],
    reference: Optional[str],
    user_answer: str,
) -> EvaluationResult:
    if is_admission_of_not_knowing(user_answer):
        from src.schemas import CriterionScore
        return EvaluationResult(
            overall_score=1,
            criterion_scores=[
                CriterionScore(
                    criterion=r,
                    met=False,
                    comment="Candidate indicated they are unfamiliar with this question."
                )
                for r in (rubric if rubric else ["Core Concept Comprehension"])
            ],
            strengths="None identified for this question.",
            improvements=(
                "It is completely fine to be unfamiliar with this concept. In real interviews, "
                "be honest when unsure, but feel free to discuss how you would troubleshoot or think through the problem. "
                "Take a moment to read the solution below."
            ),
            ideal_answer_summary=reference or "Review the fundamental algorithms, data structures, and trade-offs for this topic.",
        )

    rubric_formatted = "\n".join(f"- {r}" for r in rubric)
    last_err = None
    for model_name in AVAILABLE_MODELS:
        try:
            eval_llm = _get_llm(model=model_name).with_structured_output(EvaluationResult)
            chain = _EVALUATE_PROMPT | eval_llm
            return chain.invoke({
                "kb_question": kb_question or question_asked,
                "question_asked": question_asked,
                "rubric": rubric_formatted,
                "reference": reference or "Not provided — grade based on rubric criteria only.",
                "user_answer": user_answer,
            })
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Evaluation failed across all model failovers.")


def pick_kb_entry_for_topic(
    raw_entries: List[Dict],
    topic: Topic,
    difficulty: str,
    asked_ids: List[str],
) -> Optional[Dict]:
    candidates = [
        e for e in raw_entries
        if e["topic"] == topic.value and e["id"] not in asked_ids
    ]
    if not candidates:
        return None

    exact = [e for e in candidates if e["difficulty"] == difficulty]
    pool = exact if exact else candidates
    return random.choice(pool)


TOPIC_KEYWORDS = {
    Topic.ARRAYS: ["array", "arrays", "data structure", "dsa", "algorithm", "python", "java", "c++", "golang", "backend"],
    Topic.LINKED_LISTS: ["linked list", "pointer", "data structure", "dsa", "algorithm"],
    Topic.TREES: ["tree", "binary tree", "bst", "heap", "trie", "graph", "dsa", "algorithm"],
    Topic.DYNAMIC_PROGRAMMING: ["dynamic programming", "dp", "recursion", "memoization", "algorithm", "optimization"],
    Topic.GRAPHS: ["graph", "bfs", "dfs", "dijkstra", "topological", "network", "node"],
    Topic.OOPS: ["oops", "object oriented", "oop", "class", "design pattern", "inheritance", "polymorphism", "solid", "software"],
    Topic.OPERATING_SYSTEMS: ["operating system", "os", "thread", "process", "concurrency", "linux", "memory", "kernel"],
    Topic.SYSTEM_DESIGN: ["system design", "microservice", "distributed", "scalab", "api", "redis", "kafka", "database", "sql", "architecture", "backend"],
    Topic.PYTHON: ["python", "django", "flask", "fastapi", "pandas", "numpy", "scripting"],
    Topic.MACHINE_LEARNING: ["machine learning", "deep learning", "neural network", "scikit", "tensorflow", "pytorch", "nlp", "computer vision", "data science", "llm", "ai model"],
}


def pick_next_topic(
    topics_asked: List[Topic],
    jd_text: str,
    stage: str,
    resume_text: str = "",
) -> Optional[Topic]:
    if stage == "HR":
        return Topic.HR

    combined_text = f"{jd_text} {resume_text}".lower()
    technical_topics = [t for t in Topic if t != Topic.HR]
    remaining = [t for t in technical_topics if t not in topics_asked]
    if not remaining:
        return None

    ml_keywords = ["machine learning", "deep learning", "neural network", "data science", "scikit", "tensorflow", "pytorch", "nlp", "llm", "ai model"]
    has_ml = any(k in combined_text for k in ml_keywords)
    if not has_ml and Topic.MACHINE_LEARNING in remaining:
        remaining = [t for t in remaining if t != Topic.MACHINE_LEARNING]

    if not remaining:
        return None

    matched_topics = [
        t for t in remaining
        if any(kw in combined_text for kw in TOPIC_KEYWORDS.get(t, []))
    ]

    pool = matched_topics if matched_topics else remaining
    return random.choice(pool)
