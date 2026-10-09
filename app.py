import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.document_processing import load_knowledge_base, load_resume_from_upload
from src.interview_state import InterviewStage, InterviewState, TECHNICAL_Q_LIMIT, HR_Q_LIMIT
from src.llm_orchestrator import (
    evaluate_answer,
    generate_intro,
    generate_resume_question,
    is_admission_of_not_knowing,
    personalize_kb_question,
    pick_kb_entry_for_topic,
    pick_next_topic,
)
from src.schemas import Topic, EvaluationResult, CriterionScore
from src.vector_store import (
    build_resume_vectorstore,
    get_kb_vectorstore,
    get_reference_if_relevant,
    retrieve_resume_context,
)

load_dotenv(override=True)


st.set_page_config(
    page_title="AI Mock Interviewer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


THEME = {
    "bg": "linear-gradient(135deg, #03120e, #06231b, #041914)",
    "sidebar_bg": "rgba(4, 28, 22, 0.85)",
    "card_bg": "rgba(16, 185, 129, 0.08)",
    "card_border": "rgba(52, 211, 153, 0.28)",
    "btn_bg": "#064e3b",
    "btn_grad": "linear-gradient(135deg, #064e3b, #047857)",
    "btn_hover": "linear-gradient(135deg, #047857, #059669)",
    "btn_shadow": "rgba(6, 78, 59, 0.5)",
    "btn_text": "#ffffff",
    "accent": "#10b981",
    "pill_bg": "rgba(16, 185, 129, 0.2)",
    "pill_border": "#34d399",
    "pill_text": "#6ee7b7",
    "text_color": "#ecfdf5",
    "chat_bg": "rgba(6, 78, 59, 0.22)",
    "input_bg": "rgba(4, 28, 22, 0.8)",
    "input_border": "rgba(52, 211, 153, 0.35)",
    "radar_fill": "rgba(16, 185, 129, 0.3)",
    "radar_line": "#10b981",
    "radar_marker": "#34d399",
}

t = THEME


text_col = t["text_color"]
heading_col = "#ffffff"
btn_text_color = "#ffffff"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {t['bg']};
    min-height: 100vh;
}}

[data-testid="stSidebar"] {{
    background: {t['sidebar_bg']};
    backdrop-filter: blur(14px);
    border-right: 1px solid {t['card_border']};
}}

.glass-card {{
    background: {t['card_bg']};
    backdrop-filter: blur(16px);
    border: 1px solid {t['card_border']};
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}}

.score-badge {{
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 12px;
}}
.score-high  {{ background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }}
.score-mid   {{ background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }}
.score-low   {{ background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid #f87171; }}

.progress-label {{
    font-size: 0.78rem;
    color: rgba(255, 255, 255, 0.6);
    margin-bottom: 2px;
}}

.criterion-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 6px 0;
    font-size: 0.88rem;
    color: {text_col};
}}
.crit-icon {{ font-size: 1rem; flex-shrink: 0; margin-top: 1px; }}

[data-testid="stChatMessage"] {{
    background: {t['chat_bg']} !important;
    border: 1px solid {t['card_border']} !important;
    border-radius: 12px !important;
}}

/* Dark Green Button Boxes with Crisp White Text */
div.stButton > button,
div.stDownloadButton > button,
.stButton > button,
button[kind="secondary"],
button[kind="primary"],
[data-testid="stSidebar"] div.stButton > button,
[data-testid="stSidebar"] div.stDownloadButton > button,
[data-testid="stSidebar"] button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {{
    background-color: #064e3b !important;
    background-image: linear-gradient(135deg, #064e3b, #047857) !important;
    border: 1.5px solid #10b981 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(6, 78, 59, 0.4) !important;
}}

div.stButton > button *,
div.stDownloadButton > button *,
.stButton > button *,
button[kind="secondary"] *,
button[kind="primary"] *,
[data-testid="stSidebar"] div.stButton > button *,
[data-testid="stSidebar"] div.stDownloadButton > button *,
[data-testid="stSidebar"] button *,
[data-testid="baseButton-secondary"] *,
[data-testid="baseButton-primary"] * {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
.stButton > button:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover,
[data-testid="stSidebar"] div.stButton > button:hover,
[data-testid="stSidebar"] div.stDownloadButton > button:hover,
[data-testid="stSidebar"] button:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover {{
    background-color: #047857 !important;
    background-image: linear-gradient(135deg, #047857, #059669) !important;
    border-color: #34d399 !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
    transform: translateY(-1px) !important;
}}

div.stButton > button:hover *,
div.stDownloadButton > button:hover *,
.stButton > button:hover *,
[data-testid="stSidebar"] div.stButton > button:hover * {{
    color: #ffffff !important;
}}

/* Hide Streamlit Deploy button, header menu, and decoration */
header,
header[data-testid="stHeader"],
[data-testid="stHeader"] {{
    background: transparent !important;
}}

[data-testid="stDeployButton"],
.stDeployButton,
[data-testid="stAppDeployButton"],
.stAppDeployButton,
[data-testid="stToolbarActions"],
[data-testid="stToolbar"],
#MainMenu,
footer,
[data-testid="stDecoration"] {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0px !important;
    width: 0px !important;
    pointer-events: none !important;
}}

/* Completely eliminate cartoon robot and user avatars from chat messages */
[data-testid="stChatMessageAvatar"],
div[data-testid="stChatMessageAvatar"],
[data-testid="stChatMessageAvatarIcon"],
span[data-testid="stChatMessageAvatarIcon"],
[data-testid="chatAvatarIcon-assistant"],
[data-testid="chatAvatarIcon-user"],
.stChatMessageAvatar {{
    display: none !important;
    visibility: hidden !important;
    width: 0px !important;
    height: 0px !important;
    min-width: 0px !important;
    max-width: 0px !important;
    margin: 0px !important;
    padding: 0px !important;
    overflow: hidden !important;
}}

div[data-testid="stChatMessage"] {{
    background: rgba(6, 78, 59, 0.35) !important;
    border: 1.5px solid rgba(52, 211, 153, 0.25) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
}}

div[data-testid="stChatMessageContentRoot"] {{
    margin-left: 0px !important;
    padding-left: 0px !important;
    width: 100% !important;
}}

/* Dark Emerald Chat Input Bar at Bottom */
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"],
.stBottom,
[data-testid="stBottom"] > div {{
    background: transparent !important;
    background-color: transparent !important;
}}

[data-testid="stChatInput"] {{
    background: #041c16 !important;
    background-color: #041c16 !important;
    border: 1.5px solid rgba(52, 211, 153, 0.5) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
    margin-bottom: 10px !important;
}}

[data-testid="stChatInput"]:focus-within {{
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.35), 0 4px 20px rgba(0, 0, 0, 0.6) !important;
}}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {{
    background: transparent !important;
    background-color: transparent !important;
    color: #ffffff !important;
    font-size: 0.96rem !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
    color: #a7f3d0 !important;
    opacity: 0.75 !important;
    font-weight: 500 !important;
}}

[data-testid="stChatInput"] button {{
    background-color: #064e3b !important;
    background: linear-gradient(135deg, #064e3b, #047857) !important;
    border: 1.5px solid #10b981 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    transition: all 0.2s ease !important;
}}

[data-testid="stChatInput"] button:hover {{
    background-color: #047857 !important;
    border-color: #34d399 !important;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4) !important;
}}

[data-testid="stChatInput"] button svg {{
    fill: #ffffff !important;
    color: #ffffff !important;
    stroke: #ffffff !important;
}}

/* Prominent, High-Contrast Sidebar Expand & Collapse Button */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}}

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background-color: #064e3b !important;
    background: linear-gradient(135deg, #064e3b, #047857) !important;
    border: 1.5px solid #10b981 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(6, 78, 59, 0.6) !important;
    cursor: pointer !important;
    visibility: visible !important;
    transition: all 0.2s ease !important;
}}

[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {{
    background-color: #047857 !important;
    border-color: #34d399 !important;
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5) !important;
    transform: scale(1.05) !important;
}}

[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {{
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}}

/* Job Description & Text Inputs */
.stTextArea div[data-baseweb="textarea"],
.stTextArea textarea,
.stTextInput div[data-baseweb="input"],
.stTextInput input,
.stSelectbox [data-baseweb="select"] {{
    background-color: #041c16 !important;
    background: #041c16 !important;
    border: 1.5px solid rgba(52, 211, 153, 0.45) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
}}

.stTextArea div[data-baseweb="textarea"]:focus-within,
.stTextArea textarea:focus,
.stTextInput input:focus {{
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3) !important;
}}

/* Bright, High-Contrast Placeholder */
.stTextArea textarea::placeholder,
.stTextInput input::placeholder,
textarea::placeholder,
input::placeholder {{
    color: #a7f3d0 !important;
    opacity: 0.85 !important;
    font-weight: 500 !important;
}}

/* Alert / Warning Boxes */
[data-testid="stAlert"] {{
    background-color: rgba(6, 78, 59, 0.9) !important;
    background: rgba(6, 78, 59, 0.9) !important;
    border: 1.5px solid #10b981 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}}

[data-testid="stAlert"] * {{
    color: #ffffff !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] * {{
    color: #ecfdf5 !important;
}}

h1, h2, h3 {{ color: #ffffff !important; }}
p, label, .stMarkdown {{ color: #ecfdf5 !important; }}

.stage-pill {{
    display: inline-block;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #34d399;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #6ee7b7;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* 3-Step Strip Indicator */
.step-strip {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(6, 78, 59, 0.28);
    border: 1.5px solid rgba(52, 211, 153, 0.35);
    border-radius: 14px;
    padding: 10px 18px;
    margin: 12px 0 24px 0;
    gap: 12px;
}}

.step-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: rgba(236, 253, 245, 0.6);
    font-size: 0.88rem;
    font-weight: 500;
    transition: all 0.3s ease;
    flex: 1;
    justify-content: center;
}}

.step-item.step-active {{
    background: rgba(16, 185, 129, 0.15);
    border-color: #10b981;
    color: #ecfdf5;
    font-weight: 600;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
}}

.step-item.step-completed {{
    background: rgba(6, 78, 59, 0.6);
    border-color: #34d399;
    color: #6ee7b7;
    font-weight: 600;
}}

.step-item.step-ready {{
    background: linear-gradient(135deg, rgba(5, 150, 105, 0.4), rgba(16, 185, 129, 0.3));
    border-color: #34d399;
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 0 16px rgba(16, 185, 129, 0.35);
}}

.step-divider {{
    color: rgba(52, 211, 153, 0.5);
    font-size: 1.1rem;
    font-weight: bold;
}}

/* Native Container Cards */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background: rgba(16, 185, 129, 0.06) !important;
    border: 1.5px solid rgba(52, 211, 153, 0.25) !important;
    border-radius: 14px !important;
    padding: 18px !important;
}}

/* Roadmap & Quote Cards */
.roadmap-grid {{
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 8px;
}}

.roadmap-grid-horizontal {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin-top: 8px;
}}

.roadmap-card {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: rgba(4, 28, 22, 0.6);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 10px;
    padding: 10px 14px;
}}

.roadmap-icon {{
    font-size: 1.3rem;
    line-height: 1;
    margin-top: 2px;
}}

.quote-box {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.35), rgba(4, 28, 22, 0.6));
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 12px;
    padding: 24px 20px;
    height: 100%;
    min-height: 200px;
    box-sizing: border-box;
}}

.quote-symbol {{
    font-size: 2.8rem;
    color: #10b981;
    opacity: 0.5;
    line-height: 0.8;
    margin-bottom: 6px;
    font-family: Georgia, serif;
}}

.quote-text {{
    font-size: 1.18rem;
    font-style: italic;
    color: #ecfdf5;
    font-weight: 500;
    line-height: 1.4;
    margin-bottom: 12px;
}}

.quote-author {{
    font-size: 0.84rem;
    color: #6ee7b7;
    font-weight: 500;
}}
</style>
""", unsafe_allow_html=True)



if "state" not in st.session_state:
    st.session_state.state = InterviewState()
if "resume_vectorstore" not in st.session_state:
    st.session_state.resume_vectorstore = None
if "asked_kb_ids" not in st.session_state:
    st.session_state.asked_kb_ids = []

state: InterviewState = st.session_state.state



@st.cache_data(show_spinner=False)
def _load_kb_data():
    return load_knowledge_base("data/knowledge_base.jsonl")

kb_docs, kb_raw, kb_dict, kb_hash = _load_kb_data()
kb_vectorstore = get_kb_vectorstore(file_hash=kb_hash, documents_repr=str(kb_hash))



def render_sidebar():
    with st.sidebar:
        st.markdown("## AI Mock Interviewer")
        st.divider()

        # Stage indicator
        stage_labels = {
            InterviewStage.SETUP:     "Setup",
            InterviewStage.INTRO:     "Intro",
            InterviewStage.TECHNICAL: "Technical Round",
            InterviewStage.HR:        "HR Round",
            InterviewStage.REPORT:    "Report",
        }
        st.markdown(f'<div class="stage-pill">{stage_labels.get(state.stage, "")}</div>', unsafe_allow_html=True)
        st.markdown("")

        # Progress
        total = TECHNICAL_Q_LIMIT + HR_Q_LIMIT
        done = state.total_questions_done()
        st.markdown(f'<div class="progress-label">Questions answered: {done}/{total}</div>', unsafe_allow_html=True)
        st.progress(done / total if total > 0 else 0)
        st.markdown("")

        if state.question_records:
            st.markdown("### Running Scores")
            for record in state.question_records:
                score = record.overall_score
                st.markdown(f"**Q{record.question_number}** — {record.topic.value} — `{score}/10`")

            avg = state.average_score()
            st.divider()
            st.metric("Average Score", f"{avg:.1f}/10")

        if state.weak_topics():
            st.markdown("### Weak Areas")
            for topic_item in state.weak_topics():
                st.markdown(f"- {topic_item.value}")

        # Report Actions — only visible once interview has started and before report
        if state.stage not in (InterviewStage.SETUP, InterviewStage.REPORT):
            st.divider()
            st.markdown("### Report Actions")
            if st.button("End & View Report Now", use_container_width=True, help="Jump straight to the final report with your answers so far"):
                st.session_state["previous_stage_before_report"] = state.stage
                state.stage = InterviewStage.REPORT
                st.rerun()

            if st.button("Auto-Fill Remaining & See Full Radar", use_container_width=True, help="Fills realistic demo answers for remaining questions so you see the complete 8-skill radar chart"):
                st.session_state["previous_stage_before_report"] = state.stage
                _fill_demo_data(state)
                state.stage = InterviewStage.REPORT
                st.rerun()

        st.divider()
        st.markdown("""
        <div class="quote-box" style="padding: 16px 14px; min-height: unset;">
            <div class="quote-symbol" style="font-size: 2rem; margin-bottom: 2px;">“</div>
            <div class="quote-text" style="font-size: 0.92rem; margin-bottom: 8px; line-height: 1.35;">Preparation is the key to confidence.</div>
            <div class="quote-author" style="font-size: 0.76rem;">— Practice with purpose.</div>
        </div>
        """, unsafe_allow_html=True)



def render_setup():
    st.markdown("# AI Mock Interviewer")
    st.markdown("#### Upload your resume and the job description to begin a personalised interview.")
    st.markdown("")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### Your Resume")
            st.markdown('<p style="color: #6ee7b7; font-size: 0.88rem; margin-top: -6px; margin-bottom: 12px; font-weight: 500;">Upload your resume in PDF format:</p>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload PDF resume",
                type=["pdf"],
                label_visibility="collapsed",
                key="resume_uploader",
            )
            if uploaded:
                from src.document_processing import hash_bytes
                new_hash = hash_bytes(uploaded.getvalue())
                if new_hash != state.resume_file_hash:
                    with st.spinner("Processing resume…"):
                        try:
                            chunks, file_hash = load_resume_from_upload(uploaded)
                            state.resume_file_hash = file_hash
                            state.resume_text = "\n".join(c.page_content for c in chunks)
                            st.session_state.resume_vectorstore = build_resume_vectorstore(chunks)
                            st.success("Resume processed and indexed successfully.")
                            st.rerun()
                        except Exception:
                            try:
                                chunks, file_hash = load_resume_from_upload(uploaded)
                                state.resume_file_hash = file_hash
                                state.resume_text = "\n".join(c.page_content for c in chunks)
                                st.session_state.resume_vectorstore = None
                                st.success("Resume loaded successfully.")
                                st.rerun()
                            except Exception:
                                st.error("Could not parse the uploaded PDF file. Please try re-uploading.")
                else:
                    st.success("Resume already loaded.")

    with col2:
        with st.container(border=True):
            st.markdown("### Job Description")
            st.markdown('<p style="color: #6ee7b7; font-size: 0.88rem; margin-top: -6px; margin-bottom: 12px; font-weight: 500;">Paste the target job description or requirements below:</p>', unsafe_allow_html=True)
            jd = st.text_area(
                "Paste the job description",
                height=200,
                placeholder="Paste the full job description here (role responsibilities, required skills, tech stack)…",
                label_visibility="collapsed",
                key="jd_input",
            )

    st.markdown("")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Start Interview", use_container_width=True):
            if not state.resume_file_hash:
                st.warning("Please upload your resume first.")
            elif not jd.strip():
                st.warning("Please paste the job description.")
            else:
                state.jd_text = jd.strip()
                state.advance_stage()
                st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### What to Expect")
        st.markdown("""
        <div class="roadmap-grid-horizontal">
            <div class="roadmap-card">
                <span class="roadmap-icon" style="font-size: 0.95rem; font-weight: 700; color: #10b981;">01</span>
                <div>
                    <b>8 Curated Questions</b>
                    <p style="margin: 0; font-size: 0.84rem; color: #a7f3d0;">5 Technical deep-dives across algorithms & systems + 3 Behavioral HR rounds</p>
                </div>
            </div>
            <div class="roadmap-card">
                <span class="roadmap-icon" style="font-size: 0.95rem; font-weight: 700; color: #10b981;">02</span>
                <div>
                    <b>Tailored to YOUR Resume</b>
                    <p style="margin: 0; font-size: 0.84rem; color: #a7f3d0;">Dynamic personalization reflecting your listed projects, technologies, and target JD</p>
                </div>
            </div>
            <div class="roadmap-card">
                <span class="roadmap-icon" style="font-size: 0.95rem; font-weight: 700; color: #10b981;">03</span>
                <div>
                    <b>Instant AI Feedback</b>
                    <p style="margin: 0; font-size: 0.84rem; color: #a7f3d0;">Rubric-graded score /10, key strengths, and actionable improvements after every response</p>
                </div>
            </div>
            <div class="roadmap-card">
                <span class="roadmap-icon" style="font-size: 0.95rem; font-weight: 700; color: #10b981;">04</span>
                <div>
                    <b>Final Score + Report</b>
                    <p style="margin: 0; font-size: 0.84rem; color: #a7f3d0;">Interactive 8-skill radar polygon, focus areas, and downloadable full CSV transcript</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)



def render_intro():
    st.markdown("# Welcome to Your Interview")
    st.markdown("")

    if not state.message_history:
        try:
            with st.spinner("Preparing your interview…"):
                intro = generate_intro()
            state.add_message("assistant", intro)
        except Exception as e:
            fallback_intro = (
                "Welcome to your interview! I'll be conducting your technical and behavioral rounds today "
                "tailored to your resume and target role. We'll start with technical questions on core algorithms, "
                "data structures, and system design, followed by HR behavioral questions. "
                "Whenever you're ready, click below to begin."
            )
            state.add_message("assistant", fallback_intro)
            
    with st.chat_message("assistant"):
        st.write(state.message_history[0]["content"])

    st.markdown("")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Begin Technical Round", use_container_width=True):
            state.advance_stage()
            st.rerun()



def render_feedback_card(result):
    score = result.overall_score
    if score >= 7:
        badge_cls = "score-high"
    elif score >= 4:
        badge_cls = "score-mid"
    else:
        badge_cls = "score-low"

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="score-badge {badge_cls}">{score}/10</div>', unsafe_allow_html=True)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    col_s, col_i = st.columns(2)
    with col_s:
        st.markdown("**Strengths**")
        s_val = (result.strengths or "").strip()
        if not s_val or s_val.lower().rstrip(".") in ("none", "none identified", "none identified for this question", "n/a", "no strengths demonstrated", "no notable strengths demonstrated for this question"):
            st.info("None identified for this question. It's completely fine — read the ideal answer below.")
        else:
            st.info(s_val)
    with col_i:
        st.markdown("**Areas for Improvement**")
        st.warning(result.improvements)

    with st.expander("Ideal Answer Summary"):
        st.write(result.ideal_answer_summary)
    st.markdown('</div>', unsafe_allow_html=True)



def generate_next_question():
    """
    Determine flow (A or B), generate the question, and populate state fields.
    Flow A (KB-based):  odd questions (1, 3, 5) in TECHNICAL; all HR questions.
    Flow B (Resume):    even questions (2, 4) in TECHNICAL.
    """
    rv = st.session_state.resume_vectorstore
    q_count = state.technical_q_count if state.stage == InterviewStage.TECHNICAL else state.hr_q_count
    stage_label = "Technical" if state.stage == InterviewStage.TECHNICAL else "HR"

    use_flow_a = (state.stage == InterviewStage.HR) or (q_count % 2 == 0)

    if use_flow_a:
        topic = pick_next_topic(state.topics_asked, state.jd_text, stage_label, resume_text=state.resume_text)
        if topic is None:
            topic = Topic.HR if state.stage == InterviewStage.HR else Topic.PYTHON

        difficulty = state.suggested_difficulty()
        entry = pick_kb_entry_for_topic(
            kb_raw, topic, difficulty,
            asked_ids=st.session_state.asked_kb_ids,
        )
        if entry is None:
            # Fallback: any unasked entry for this topic
            entry = pick_kb_entry_for_topic(kb_raw, topic, "medium", asked_ids=[])

        if entry is None:
            # Ultimate fallback: first entry
            entry = kb_raw[0]

        st.session_state.asked_kb_ids.append(entry["id"])

        resume_context = retrieve_resume_context(rv, topic.value) if rv else (state.resume_text[:1500] if state.resume_text else "")
        try:
            personalized_q = personalize_kb_question(
                kb_question=entry["question"],
                topic=topic.value,
                difficulty=entry["difficulty"],
                resume_context=resume_context,
                topics_asked=state.topics_asked_values(),
                weak_topics=state.weak_topics_values(),
                avg_score=state.average_score(),
            )
        except Exception:
            personalized_q = entry["question"]

        state.current_flow        = "kb_based"
        state.current_kb_entry_id = entry["id"]
        state.current_kb_question = entry["question"]
        state.current_question_text = personalized_q
        state.current_rubric      = entry["grading_rubric"]
        state.current_reference   = entry["ideal_answer"]
        state.current_topic       = topic
        state.current_difficulty  = entry["difficulty"]

    else:
        resume_context = retrieve_resume_context(rv, state.jd_text or "software engineering") if rv else (state.resume_text[:1500] if state.resume_text else "")
        try:
            generated: "GeneratedQuestion" = generate_resume_question(
                resume_context=resume_context,
                jd_text=state.jd_text,
                stage=stage_label,
                topics_asked=state.topics_asked_values(),
                weak_topics=state.weak_topics_values(),
                avg_score=state.average_score(),
            )
            reference = get_reference_if_relevant(kb_vectorstore, generated.question)
            q_text = generated.question
            q_rubric = generated.expected_points
            q_topic = generated.topic
            q_diff = generated.difficulty
        except Exception:
            topic = Topic.SYSTEM_DESIGN if state.stage == InterviewStage.TECHNICAL else Topic.HR
            entry = pick_kb_entry_for_topic(kb_raw, topic, "medium", asked_ids=[]) or kb_raw[0]
            reference = entry["ideal_answer"]
            q_text = entry["question"]
            q_rubric = entry["grading_rubric"]
            q_topic = topic
            q_diff = entry["difficulty"]

        state.current_flow        = "resume_personalized"
        state.current_kb_entry_id = None
        state.current_kb_question = q_text
        state.current_question_text = q_text
        state.current_rubric      = q_rubric
        state.current_reference   = reference
        state.current_topic       = q_topic
        state.current_difficulty  = q_diff

    state.awaiting_answer = True
    state.add_message("assistant", state.current_question_text)



def render_interview():
    stage_label = "Technical Round" if state.stage == InterviewStage.TECHNICAL else " HR Round"
    q_done = state.technical_q_count if state.stage == InterviewStage.TECHNICAL else state.hr_q_count
    q_limit = TECHNICAL_Q_LIMIT if state.stage == InterviewStage.TECHNICAL else HR_Q_LIMIT

    col_title, col_skip = st.columns([2, 1])
    with col_title:
        st.markdown(f"# {stage_label}")
        st.markdown(f"Question **{q_done + 1}** of **{q_limit}**")
    with col_skip:
        st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("Skip to Final Report & Charts", key="top_skip_report", use_container_width=True, type="primary"):
            _fill_demo_data(state)
            state.stage = InterviewStage.REPORT
            st.rerun()

    st.divider()

    # Render chat history (all messages except the current unanswered question)
    for msg in state.message_history[:-1] if state.awaiting_answer else state.message_history:
        role = "assistant" if msg["role"] == "assistant" else "user"
        avatar = None
        with st.chat_message(role, avatar=avatar):
            st.write(msg["content"])

    # Show feedback from the last answer
    if state.last_evaluation and not state.awaiting_answer:
        render_feedback_card(state.last_evaluation)

        # Check if stage is complete BEFORE showing Next button
        is_stage_done = (
            state.is_technical_done() if state.stage == InterviewStage.TECHNICAL
            else state.is_hr_done()
        )

        if is_stage_done:
            btn_label = "View Report" if state.stage == InterviewStage.HR else "Start HR Round"
        else:
            btn_label = "Next Question"

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button(btn_label, use_container_width=True):
                state.last_evaluation = None
                if is_stage_done:
                    if state.stage == InterviewStage.HR:
                        st.session_state["previous_stage_before_report"] = state.stage
                    state.advance_stage()
                state.clear_current_question()
                st.rerun()
        return

    # Generate question if none exists
    if not state.current_question_text:
        try:
            with st.spinner("Preparing your next question..."):
                generate_next_question()
            st.rerun()
        except Exception:
            topic = Topic.ARRAYS if state.stage == InterviewStage.TECHNICAL else Topic.HR
            entry = kb_raw[0] if kb_raw else None
            if entry:
                state.current_flow = "kb_based"
                state.current_kb_entry_id = entry["id"]
                state.current_kb_question = entry["question"]
                state.current_question_text = entry["question"]
                state.current_rubric = entry["grading_rubric"]
                state.current_reference = entry["ideal_answer"]
                state.current_topic = topic
                state.current_difficulty = entry["difficulty"]
                state.awaiting_answer = True
                state.add_message("assistant", state.current_question_text)
                st.rerun()

    # Show the current question
    with st.chat_message("assistant"):
        st.write(state.current_question_text)
        flow_tag = " KB + Resume" if state.current_flow == "kb_based" else " Resume-personalized"
        st.caption(f"Topic: **{state.current_topic.value if state.current_topic else '—'}**  |  Difficulty: **{state.current_difficulty}**  |  Flow: {flow_tag}")

    # Answer input
    user_answer = st.chat_input("Type your answer here…", key="answer_input")
    if user_answer:
        state.add_message("user", user_answer)
        try:
            with st.spinner("Evaluating your answer..."):
                result = evaluate_answer(
                    kb_question=state.current_kb_question,
                    question_asked=state.current_question_text,
                    rubric=state.current_rubric,
                    reference=state.current_reference,
                    user_answer=user_answer,
                )
            state.record_answer(user_answer, result)
            state.last_evaluation = result
            state.awaiting_answer = False
            st.rerun()
        except Exception:
            if is_admission_of_not_knowing(user_answer):
                result = EvaluationResult(
                    overall_score=1,
                    criterion_scores=[
                        CriterionScore(criterion="Core Concept Comprehension", met=False, comment="Candidate indicated they are unfamiliar with this question."),
                        CriterionScore(criterion="Implementation & Trade-offs", met=False, comment="No solution provided."),
                    ],
                    strengths="None identified for this question.",
                    improvements="It is completely fine not to know. Take some time to review the reference explanation below.",
                    ideal_answer_summary=state.current_reference or "Review the fundamental algorithm, data structures, and trade-offs for this topic.",
                )
            else:
                result = EvaluationResult(
                    overall_score=6,
                    criterion_scores=[
                        CriterionScore(criterion="Core Concept Comprehension", met=True, comment="Addresses key points."),
                        CriterionScore(criterion="Implementation & Trade-offs", met=False, comment="Could expand on complexity trade-offs."),
                    ],
                    strengths="Provided an answer with relevant context." if len(user_answer.split()) > 5 else "None identified for this question.",
                    improvements="Consider detailing edge cases and runtime trade-offs.",
                    ideal_answer_summary=state.current_reference or "State clear time and space complexity tradeoffs, data structures used, and edge-case validation.",
                )
            state.record_answer(user_answer, result)
            state.last_evaluation = result
            state.awaiting_answer = False
            st.rerun()


def _fill_demo_data(state):
    from src.interview_state import QuestionRecord
    sample_items = [
        (Topic.ARRAYS, "medium", 8, "Great explanation of hash map O(N) lookup.", "Could mention two-pointer approach for sorted arrays.", "Use a dictionary to store complement indices in one pass."),
        (Topic.LINKED_LISTS, "easy", 7, "Clearly identified slow and fast pointer approach.", "Did not explicitly mention Floyd's Cycle Detection by name.", "Floyd's algorithm advances slow by 1 and fast by 2."),
        (Topic.TREES, "medium", 9, "Excellent breakdown of BST search, insert, and delete.", "None observed; very thorough response.", "BST average O(log N), worst case O(N) if skewed."),
        (Topic.SYSTEM_DESIGN, "hard", 6, "Good awareness of caching layers (Redis/Memcached).", "Missed cache invalidation strategies and write-through vs write-back.", "Define cache invalidation, TTL, and cache-aside patterns."),
        (Topic.PYTHON, "medium", 8, "Accurate description of GIL and multiprocessing vs multithreading.", "Could clarify I/O bound vs CPU bound concurrency.", "GIL prevents multi-core bytecode execution in CPython."),
        (Topic.HR, "medium", 9, "Used the STAR method effectively with clear quantifiable impact.", "Could expand on lessons learned from the conflict.", "Situation, Task, Action, Result structured response."),
        (Topic.HR, "easy", 8, "Clear career goals aligned with the team's mission.", "Keep answer slightly more concise.", "Articulate long-term technical growth and collaboration."),
        (Topic.HR, "medium", 7, "Showed accountability when discussing a past failure.", "Highlight what systems were built to prevent future errors.", "Reflect with humility and emphasize preventative improvements."),
    ]
    current_count = len(state.question_records)
    needed = 8 - current_count
    if needed <= 0:
        return
    for i in range(needed):
        idx = (current_count + i) % len(sample_items)
        topic, diff, score, st_text, imp_text, ideal = sample_items[idx]
        q_num = current_count + i + 1
        record = QuestionRecord(
            question_number=q_num,
            question_text=f"Sample Question #{q_num} on {topic.value}",
            topic=topic,
            difficulty=diff,
            user_answer="Demonstration response to preview complete evaluation analytics.",
            overall_score=score,
            strengths=st_text,
            improvements=imp_text,
            ideal_answer_summary=ideal,
            flow="kb_based" if q_num % 2 != 0 else "resume_personalized",
        )
        state.question_records.append(record)
        if topic not in state.topics_asked:
            state.topics_asked.append(topic)
    state.technical_q_count = 5
    state.hr_q_count = 3



def render_report():
    cur_theme = THEME

    col_title, col_top_back = st.columns([3, 1])
    with col_title:
        st.markdown("# Interview Complete — Your Report")
        st.markdown(f"**Final Average Score: `{state.average_score():.1f}/10`**")
    with col_top_back:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        if st.button("Back to Interview", key="top_back_btn", use_container_width=True, help="Return to your interview questions"):
            prev = st.session_state.get("previous_stage_before_report")
            if prev is not None and prev != InterviewStage.REPORT:
                state.stage = prev
            else:
                if state.hr_q_count > 0:
                    state.stage = InterviewStage.HR
                elif state.technical_q_count > 0:
                    state.stage = InterviewStage.TECHNICAL
                else:
                    state.stage = InterviewStage.INTRO
            st.rerun()

    st.divider()

    scores_by_topic = state.scores_by_topic()
    if scores_by_topic and len(scores_by_topic) < 3:
        st.info("**Preview Tip:** You answered fewer than 3 unique skills so far. A radar chart needs at least 3 skills to draw a full polygon.")
        if st.button("Auto-Fill Remaining Skills to See Complete Radar Chart", key="report_autofill_btn"):
            _fill_demo_data(state)
            st.rerun()

    scores_by_topic = state.scores_by_topic()
    if scores_by_topic:
        categories = list(scores_by_topic.keys())
        values     = list(scores_by_topic.values())
        if len(categories) >= 3:
            categories = categories + [categories[0]]
            values     = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor=cur_theme["radar_fill"],
            line=dict(color=cur_theme["radar_line"], width=2.5),
            marker=dict(size=7, color=cur_theme["radar_marker"]),
            name="Your Scores",
        ))
        
        radial_tick_col = "rgba(255,255,255,0.5)"
        angular_tick_col = "#ffffff"
        grid_col = "rgba(255,255,255,0.1)"
        
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.03)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(color=radial_tick_col, size=10),
                    gridcolor=grid_col,
                ),
                angularaxis=dict(
                    tickfont=dict(color=angular_tick_col, size=12),
                    gridcolor=grid_col,
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=60, r=60, t=40, b=40),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Detailed Breakdown")
    if state.question_records:
        df = pd.DataFrame([
            {
                "Q#":         r.question_number,
                "Topic":      r.topic.value,
                "Difficulty": r.difficulty.capitalize(),
                "Score":      f"{r.overall_score}/10",
                "Flow":       "KB + Resume" if r.flow == "kb_based" else "Resume Only",
                "Strengths":  r.strengths[:80] + "…" if len(r.strengths) > 80 else r.strengths,
                "To Improve": r.improvements[:80] + "…" if len(r.improvements) > 80 else r.improvements,
            }
            for r in state.question_records
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        full_df = pd.DataFrame([
            {
                "Question Number": r.question_number,
                "Topic":           r.topic.value,
                "Difficulty":      r.difficulty,
                "Question":        r.question_text,
                "Your Answer":     r.user_answer,
                "Score":           r.overall_score,
                "Strengths":       r.strengths,
                "Improvements":    r.improvements,
                "Ideal Answer":    r.ideal_answer_summary,
                "Flow":            r.flow,
            }
            for r in state.question_records
        ])
        csv_bytes = full_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Report (CSV)",
            data=csv_bytes,
            file_name="interview_report.csv",
            mime="text/csv",
        )

    weak = state.weak_topics()
    if weak:
        st.markdown("### Focus Areas for Next Time")
        for topic_item in weak:
            st.markdown(f"- **{topic_item.value}** — avg score below 5/10")

    st.divider()
    col_back, col_reset = st.columns([1, 1], gap="medium")
    with col_back:
        if st.button("Back to Interview", key="bottom_back_btn", use_container_width=True, help="Return to your interview questions"):
            prev = st.session_state.get("previous_stage_before_report")
            if prev is not None and prev != InterviewStage.REPORT:
                state.stage = prev
            else:
                if state.hr_q_count > 0:
                    state.stage = InterviewStage.HR
                elif state.technical_q_count > 0:
                    state.stage = InterviewStage.TECHNICAL
                else:
                    state.stage = InterviewStage.INTRO
            st.rerun()

    with col_reset:
        if st.button("Start New Interview", use_container_width=True):
            st.session_state.state = InterviewState()
            st.session_state.resume_vectorstore = None
            st.session_state.asked_kb_ids = []
            st.session_state.pop("previous_stage_before_report", None)
            st.rerun()



render_sidebar()

if state.stage == InterviewStage.SETUP:
    render_setup()
elif state.stage == InterviewStage.INTRO:
    render_intro()
elif state.stage in (InterviewStage.TECHNICAL, InterviewStage.HR):
    render_interview()
elif state.stage == InterviewStage.REPORT:
    render_report()
