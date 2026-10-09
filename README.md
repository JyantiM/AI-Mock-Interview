# AI Mock Interviewer

## Overview
<sub>AI Mock Interviewer simulates technical and behavioral hiring loops by assessing candidate knowledge across core computer science concepts, system architecture, and situational judgment. It implements dual question generation via vector similarity search and dynamic resume synthesis with real-time rubric-based evaluation.</sub>
---
## Core Features
- **Two-Stage Interview Pipeline**:
  - **Technical Round**: 5 questions spanning algorithms, system architecture, and domain-specific engineering principles.
  - **HR & Behavioral Round**: 3 questions assessing collaboration, conflict resolution, ownership, and communication.
- **Dual Question Generation Strategy**:
  - **Knowledge Base Retrieval (Flow A)**: Queries standardized interview questions and rubrics from a FAISS vector index and dynamically personalizes them using candidate resume details.
  - **Dynamic Resume Synthesis (Flow B)**: Generates questions directly from candidate projects and target job descriptions, automatically constructing matching 3 to 5 point evaluation rubrics.
- **Adaptive Difficulty Engine**: Adjusts subsequent question difficulty between Easy, Medium, and Hard based on running candidate evaluation scores.
- **Structured Rubric Evaluation**: Scores responses from 0 to 10 against predefined criteria with boolean satisfaction checks, identified strengths, areas for improvement, and ideal reference answers.
- **Skip & Pass Handling**: Includes dedicated handling for unfamiliar topics, awarding transparent feedback and full reference explanations without interrupting the interview loop.
- **Analytics & Export**:
  - Interactive radar chart visualizing candidate competency across tested topics.
  - Comprehensive final report table detailing question-by-question outcomes.
  - One-click CSV export containing full interview transcripts, rubric criteria, candidate answers, and model solutions.
- **Document Processing**: Extracts and processes candidate resumes from PDF uploads and parses target job descriptions for semantic retrieval.
---
## Supported Topics
- **Data Structures & Algorithms**: Arrays, Linked Lists, Trees, Dynamic Programming, Graphs
- **Software Engineering & Architecture**: OOPs, Operating Systems, System Design, Python, Machine Learning
- **Behavioral**: HR & Situational Judgment
---
## Tech Stack
- **Application Framework**: Python 3.10+
- **Frontend / Interface**: Streamlit
- **LLM Orchestration**: LangChain, LangChain Core (LCEL)
- **Foundation Model**: Google Gemini (`gemini-2.5-flash` via `langchain-google-genai`)
- **Embeddings**: Google Generative AI Embeddings (`models/gemini-embedding-001`)
- **Vector Search**: FAISS (`faiss-cpu`)
- **Document Ingestion**: pypdf
- **Data Validation & Schemas**: Pydantic v2
- **Analytics & Visualization**: Plotly, Pandas
---
## Directory Structure

```text
ai_mock_interviewer/
│
├── app.py                      # Main Streamlit application and UI state router
├── requirements.txt            # Project dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules for virtual environments and cache
├── README.md                   # Project documentation
│
├── data/
│   └── knowledge_base.jsonl    # Curated interview questions, topics, and rubrics
│
└── src/
    ├── __init__.py             # Package initializer
    ├── schemas.py              # Pydantic models and topic/difficulty enums
    ├── interview_state.py      # Session state machine and progress tracking
    ├── document_processing.py  # PDF resume parser and JSONL loader
    ├── vector_store.py         # FAISS vector database initialization and retrieval
    └── llm_orchestrator.py     # Prompt chains, question generators, and rubric evaluators
```

