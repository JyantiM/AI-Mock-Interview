# 🎯 AI Mock Interviewer

An intelligent, adaptive technical & HR mock interviewer powered by **LangChain**, **FAISS**, and **Google Gemini** (100% free API tier).

---

## 🌟 Key Features

1. **Dual Question Generation Strategy**:
   - **Flow A (KB Direct Lookup + Resume Personalization)**: Fetches standardized questions and rubrics from a curated knowledge base (DSA, OOPs, System Design, etc.) and dynamically weaves in candidate project context from their resume.
   - **Flow B (Dynamic Resume Questions)**: The LLM generates tailored technical questions based directly on the candidate's resume and job description, simultaneously outputting a matching 3-5 point rubric (`expected_points`).
2. **Adaptive Difficulty**:
   - Dynamically scales difficulty between `easy`, `medium`, and `hard` based on running candidate performance.
3. **Structured Rubric Evaluation**:
   - Zero branching evaluator: tests candidates against each criterion with boolean pass/fail and constructive feedback.
4. **Performance Radar Chart & Downloadable Summary**:
   - Interactive Plotly polar/radar chart showing strengths across all tested skill domains.
   - One-click export to CSV containing full questions, candidate responses, strengths, improvements, and ideal answers.
5. **Modern Dark UI**:
   - Sleek glassmorphism aesthetic built with Streamlit.

---

## 🚀 Quick Start Guide

### 1. Get a Free Gemini API Key
- Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
- Sign in with any Google account and click **Create API Key**.
- No credit card required.

### 2. Configure `.env`
Open `.env` in this directory and paste your key:
```ini
GEMINI_API_KEY=AIzaSy...
```
*(Alternatively, you can paste the key directly into the app's sidebar when launched!)*

### 3. Launch the Application
Run in your PowerShell / terminal:
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 📂 Project Architecture

```
ai_mock_interviewer/
├── .env                      # API keys (GEMINI_API_KEY)
├── .env.example              # Template
├── requirements.txt          # Python dependencies
├── app.py                    # Streamlit frontend & stage router
├── README.md                 # Project documentation
├── data/
│   └── knowledge_base.jsonl  # 25 verified interview questions & rubrics
└── src/
    ├── __init__.py
    ├── schemas.py             # Pydantic schemas (Topic enum, Difficulty, etc.)
    ├── interview_state.py     # Pure Python state machine & session tracking
    ├── document_processing.py # PDF chunking & JSONL loader
    ├── vector_store.py        # Dual FAISS index management
    └── llm_orchestrator.py    # LCEL prompt chains & Gemini integration
```
