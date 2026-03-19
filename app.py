import os

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google import genai

from extractor import extract_text, extract_project_data
from scorer import score_project
from qa import answer_question
from report import generate_stakeholder_report

# ── Environment & Gemini setup ────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuSense — Document Intelligence System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ══════════════════════════════════════════════════
   GLOBAL
══════════════════════════════════════════════════ */
.stApp { background-color: #F4F7FB; }
.block-container { padding-top: 1.5rem !important; }

/* Force all base text to be dark and readable on light bg */
p, span, div, li, label { color: #1B3A6B; }

/* ══════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════ */
[data-testid="stSidebar"] { background-color: #1B3A6B !important; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] div { color: #B8C9E1 !important; }
[data-testid="stSidebar"] hr { border-color: #2E4F82 !important; }

/* ══════════════════════════════════════════════════
   HEADINGS
══════════════════════════════════════════════════ */
h1 { color: #1B3A6B !important; font-size: 2rem !important; }
h2 { color: #1B3A6B !important; }
h3 { color: #2E86AB !important; margin-top: 0.5rem !important; }

/* ══════════════════════════════════════════════════
   SECTION HEADER BAND
══════════════════════════════════════════════════ */
.section-header {
    background: linear-gradient(90deg, #1B3A6B 0%, #2E86AB 100%);
    color: #FFFFFF !important;
    padding: 10px 20px;
    border-radius: 8px;
    margin: 24px 0 16px 0;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.4px;
}

/* ══════════════════════════════════════════════════
   TABS  (the Upload / Paste toggle)
══════════════════════════════════════════════════ */
/* Tab button bar */
.stTabs [data-baseweb="tab-list"] {
    background-color: #EEF2F8;
    border-radius: 8px;
    padding: 4px 6px;
    gap: 6px;
    border: none;
}
/* Tab BUTTON — unselected */
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #1B3A6B !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 6px 20px !important;
}
/* Tab BUTTON — selected only (role=tab scopes to the button, not the panel) */
.stTabs [role="tab"][aria-selected="true"] {
    background-color: #1B3A6B !important;
    color: #FFFFFF !important;
}
/* Remove default underline/border indicators */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
/* Tab PANEL — always white, text always dark */
.stTabs [data-baseweb="tab-panel"],
.stTabs [role="tabpanel"] {
    background-color: #FFFFFF !important;
    padding: 16px 4px 4px 4px !important;
    border-radius: 0 0 8px 8px !important;
    color: #1B3A6B !important;
}

/* ══════════════════════════════════════════════════
   FILE UPLOADER — white bg, dark text, blue border
══════════════════════════════════════════════════ */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > section,
[data-testid="stFileUploadDropzone"] {
    background-color: #FFFFFF !important;
}
[data-testid="stFileUploader"] > section,
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #2E86AB !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] div {
    color: #1B3A6B !important;
    background-color: transparent !important;
}
[data-testid="stFileUploader"] button {
    background-color: #2E86AB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
}

/* ══════════════════════════════════════════════════
   TEXT AREA (Paste tab) — white bg, dark text
══════════════════════════════════════════════════ */
[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #1B3A6B !important;
    border: 2px solid #2E86AB !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    caret-color: #1B3A6B !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #94A8C0 !important;
    opacity: 1 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #1B3A6B !important;
    box-shadow: 0 0 0 2px rgba(27,58,107,0.2) !important;
}

/* ══════════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════════ */
/* All buttons: dark text by default so they're never invisible */
.stButton > button {
    color: #1B3A6B !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
/* Primary button */
.stButton > button[kind="primary"] {
    background-color: #2E86AB !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 0.45rem 1.4rem !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1B3A6B !important;
    color: #FFFFFF !important;
}
/* Suggested question buttons */
.stButton > button[kind="secondary"] {
    background-color: #EEF4FA !important;
    border: 1px solid #C8D9EC !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #1B3A6B !important;
    color: #FFFFFF !important;
    border-color: #1B3A6B !important;
}

/* ══════════════════════════════════════════════════
   DOWNLOAD BUTTON
══════════════════════════════════════════════════ */
[data-testid="stDownloadButton"] button,
[data-testid="stDownloadButton"] button:link,
[data-testid="stDownloadButton"] button:visited {
    background-color: #2E86AB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: #1B3A6B !important;
    color: #FFFFFF !important;
}

/* ══════════════════════════════════════════════════
   STATUS BADGE
══════════════════════════════════════════════════ */
.status-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.3px;
}
.status-on-track  { background: #C8E6C9; color: #1B5E20 !important; }
.status-at-risk   { background: #FFE0B2; color: #BF360C !important; }
.status-completed { background: #BBDEFB; color: #0D47A1 !important; }
.status-unknown   { background: #ECEFF1; color: #546E7A !important; }

/* ══════════════════════════════════════════════════
   EXTRACTION INFO PANELS
══════════════════════════════════════════════════ */
.info-panel {
    background: #FFFFFF;
    border: 1px solid #DDE6F0;
    border-radius: 10px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(27,58,107,0.06);
    margin-bottom: 12px;
}
.info-panel h4 { margin: 0 0 10px 0; color: #1B3A6B !important; font-size: 15px; }
.info-panel li { line-height: 1.7; font-size: 14px; color: #333 !important; }

/* ══════════════════════════════════════════════════
   SCORECARD COLOUR CARDS
══════════════════════════════════════════════════ */
.score-card {
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-left: 5px solid;
}
.score-card-green { background: #E8F5E9; border-color: #2E7D32; }
.score-card-amber { background: #FFF8E1; border-color: #F9A825; }
.score-card-red   { background: #FFEBEE; border-color: #C62828; }
.score-card-label { font-weight: 700; font-size: 14px; color: #1B3A6B !important; }
.score-card-score { font-size: 24px; font-weight: 800; }
.score-card-just  { font-size: 12px; color: #555 !important; margin-top: 4px; line-height: 1.5; }

/* ══════════════════════════════════════════════════
   CHAT BUBBLES
══════════════════════════════════════════════════ */
.chat-wrap { max-height: 460px; overflow-y: auto; padding: 4px 2px; }
.chat-user {
    background: #DCE9F5;
    border-radius: 14px 14px 4px 14px;
    padding: 10px 14px;
    margin: 8px 0 8px auto;
    max-width: 78%;
    font-size: 14px;
    text-align: right;
    color: #1B3A6B !important;
}
.chat-ai {
    background: #FFFFFF;
    border: 1px solid #DDE6F0;
    border-radius: 14px 14px 14px 4px;
    padding: 10px 14px;
    margin: 8px auto 8px 0;
    max-width: 82%;
    font-size: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    white-space: pre-wrap;
    color: #1B3A6B !important;
}
.chat-label { font-size: 11px; font-weight: 700; color: #888 !important; margin-bottom: 4px; }

/* Chat input box */
[data-testid="stChatInput"] textarea {
    color: #1B3A6B !important;
    background-color: #FFFFFF !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #AAAAAA !important;
    opacity: 1 !important;
}

/* ══════════════════════════════════════════════════
   REPORT BOX
══════════════════════════════════════════════════ */
.report-box {
    background: #FFFFFF;
    border: 1px solid #DDE6F0;
    border-radius: 10px;
    padding: 28px 32px;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 14.5px;
    line-height: 1.85;
    color: #222 !important;
    white-space: pre-wrap;
    box-shadow: 0 2px 10px rgba(27,58,107,0.08);
}

/* ══════════════════════════════════════════════════
   META PILLS (funder / org tags)
══════════════════════════════════════════════════ */
.meta-pill {
    display: inline-block;
    background: #EEF4FA;
    border: 1px solid #C8D9EC;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 13px;
    margin-right: 8px;
    color: #1B3A6B !important;
}

/* ══════════════════════════════════════════════════
   STREAMLIT ALERTS (error / warning / success)
   — force their text to stay legible
══════════════════════════════════════════════════ */
[data-testid="stAlert"] p { color: inherit !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:24px 0 10px;">
            <div style="font-size:40px;">🔬</div>
            <div style="font-size:22px; font-weight:800; letter-spacing:1.5px;">DocuSense</div>
            <div style="font-size:11px; color:#8BA8CC; margin-top:4px; letter-spacing:0.5px;">
                DOCUMENT INTELLIGENCE SYSTEM
            </div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Supported Document Types**")
    st.markdown(
        """
        <div style="font-size:13px; color:#B8C9E1; line-height:2.1;">
        📄 &nbsp;Project Reports<br>
        📋 &nbsp;Evaluation Documents<br>
        🏛️ &nbsp;Programme Assessments<br>
        📊 &nbsp;Annual Reports<br>
        📝 &nbsp;M&amp;E Documents
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Features**")
    st.markdown(
        """
        <div style="font-size:13px; color:#B8C9E1; line-height:2.2;">
        📤 &nbsp;Upload &amp; Extract<br>
        📊 &nbsp;Health Scorecard<br>
        💬 &nbsp;Ask the Document<br>
        📝 &nbsp;Export Stakeholder Report
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    if GEMINI_API_KEY:
        st.markdown(
            "<div style='font-size:12px; color:#81C784;'>✅ &nbsp;Gemini API connected</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:12px; color:#EF9A9A;'>❌ &nbsp;GEMINI_API_KEY missing in .env</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="margin-top:40px; font-size:10px; color:#546E7A; text-align:center; line-height:1.8;">
            <br>
            DocuSense v1.0<br>
            <span style="color:#2E4F82;">AI-Powered Document Intelligence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="padding:8px 0 4px;">
    <h1 style="margin:0;">🔬 DocuSense</h1>
    <p style="color:#6B7C93; font-size:14px; margin:4px 0 0;">
        Document Intelligence System &nbsp;·&nbsp;
        <span style="color:#2E86AB; font-weight:600;">
            Powered by AI · Turn any document into actionable intelligence
        </span>
    </p>
</div>
<hr style="border-color:#DDE6F0; margin:14px 0 20px;">
""",
    unsafe_allow_html=True,
)

# ── API key guard ─────────────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    st.error(
        "**Gemini API key not found.** "
        "Please add `GEMINI_API_KEY=your_key_here` to the `.env` file "
        "in the project root directory, then restart the app."
    )
    st.info(
        "Don't have a key? Get one free at "
        "[Google AI Studio](https://aistudio.google.com/app/apikey).",
        icon="ℹ️",
    )
    st.stop()

# ── Session state initialisation ──────────────────────────────────────────────
for key in ("document_text", "extracted_data", "scores", "report_text"):
    if key not in st.session_state:
        st.session_state[key] = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================================================================
# FEATURE 1 — UPLOAD & EXTRACT
# =============================================================================
st.markdown(
    '<div class="section-header">📤 Step 1 — Upload & Extract Project Data</div>',
    unsafe_allow_html=True,
)

raw_document_text = None

tab_upload, tab_paste = st.tabs(["📤 Upload a File", "📋 Paste Text"])

with tab_upload:
    st.markdown(
        "<p style='color:#6B7C93; font-size:13px; margin-bottom:10px;'>"
        "Accepts PDF, Word (.docx), or plain text (.txt) — "
        "e.g. project reports, evaluation documents, programme assessments.</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drop your project document here",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        file_bytes = uploaded_file.read()
        with st.spinner("Reading document..."):
            try:
                raw_document_text = extract_text(file_bytes, uploaded_file.name)
            except ValueError as exc:
                st.error(str(exc))
                raw_document_text = None
            except Exception as exc:
                st.error(
                    f"Could not read **{uploaded_file.name}**. "
                    f"The file may be corrupted or password-protected.\n\n`{exc}`"
                )
                raw_document_text = None
        if raw_document_text:
            st.success(
                f"**{uploaded_file.name}** loaded — "
                f"{len(raw_document_text):,} characters extracted."
            )
        elif raw_document_text is not None:
            st.warning("The document appears to be empty or could not be read.")

with tab_paste:
    st.markdown(
        "<p style='color:#6B7C93; font-size:13px; margin-bottom:6px;'>"
        "Paste text from any project report, M&amp;E document, or evaluation.</p>",
        unsafe_allow_html=True,
    )
    pasted_text = st.text_area(
        "Paste document text",
        height=220,
        placeholder=(
            "Paste the content of a project report, evaluation document, "
            "programme assessment, annual report, or M&E document here..."
        ),
        label_visibility="collapsed",
    )
    if pasted_text and pasted_text.strip():
        raw_document_text = pasted_text.strip()

col_btn, _ = st.columns([1, 4])
with col_btn:
    analyse_btn = st.button(
        "🔍 Analyse Document",
        type="primary",
        disabled=(not raw_document_text),
        use_container_width=True,
    )

if analyse_btn and raw_document_text:
    # Reset downstream state on new analysis
    st.session_state.document_text = raw_document_text
    st.session_state.extracted_data = None
    st.session_state.scores = None
    st.session_state.chat_history = []
    st.session_state.report_text = None

    with st.spinner("🤖 Analysing document with Gemini AI — this may take a moment..."):
        try:
            st.session_state.extracted_data = extract_project_data(
                raw_document_text, gemini_client
            )
        except Exception as exc:
            st.error(
                f"**Extraction failed.** Gemini returned an unexpected response. "
                f"Please try again.\n\n`{exc}`"
            )
            st.stop()

    with st.spinner("📊 Calculating project health scores..."):
        try:
            st.session_state.scores = score_project(
                st.session_state.extracted_data, gemini_client
            )
        except Exception as exc:
            st.warning(
                f"Health scoring could not be completed: `{exc}`. "
                "You can still use the other features."
            )

# ── Display extracted data ────────────────────────────────────────────────────
if st.session_state.extracted_data:
    d = st.session_state.extracted_data

    status = d.get("project_status", "Unknown")
    status_class = {
        "On Track": "status-on-track",
        "At Risk": "status-at-risk",
        "Completed": "status-completed",
        "Unknown": "status-unknown",
    }.get(status, "status-unknown")

    # Project title + status
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown(f"### {d.get('project_name', 'Project Name Unknown')}")
    with col_status:
        st.markdown(
            f"<div style='text-align:right; padding-top:10px;'>"
            f"<span class='status-badge {status_class}'>⬤ {status}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Funder / Implementing org pills
    funder = d.get("funder", "Unknown")
    impl_org = d.get("implementing_organization", "Unknown")
    st.markdown(
        f"<div style='margin-bottom:18px;'>"
        f"<span class='meta-pill'>💰 {funder}</span>"
        f"<span class='meta-pill'>🏢 {impl_org}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Four sections in a 2×2 grid
    col_left, col_right = st.columns(2)

    def _render_list(items: list, empty_msg: str = "None identified in document."):
        if items:
            return "".join(f"<li>{item}</li>" for item in items)
        return f"<li><em style='color:#888;'>{empty_msg}</em></li>"

    with col_left:
        st.markdown(
            f"""
            <div class="info-panel">
                <h4>🎯 Project Objectives</h4>
                <ul>{_render_list(d.get('objectives', []))}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="info-panel">
                <h4>✅ Key Achievements</h4>
                <ul>{_render_list(d.get('achievements', []))}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f"""
            <div class="info-panel">
                <h4>⚠️ Challenges &amp; Risks</h4>
                <ul>{_render_list(d.get('challenges', []))}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="info-panel">
                <h4>💡 Recommendations</h4>
                <ul>{_render_list(d.get('recommendations', []))}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# FEATURE 2 — HEALTH SCORECARD
# =============================================================================
if st.session_state.scores:
    scores = st.session_state.scores

    st.markdown(
        '<div class="section-header">📊 Step 2 — Project Health Scorecard</div>',
        unsafe_allow_html=True,
    )

    col_radar, col_cards = st.columns([1, 1])

    with col_radar:
        labels = ["Delivery", "Impact", "Risk Level", "Efficiency"]
        values = [
            scores["delivery"]["score"],
            scores["impact"]["score"],
            scores["risk_level"]["score"],
            scores["efficiency"]["score"],
        ]
        # Close the polygon
        labels_closed = labels + [labels[0]]
        values_closed = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
                fillcolor="rgba(46,134,171,0.18)",
                line=dict(color="#2E86AB", width=2.5),
                marker=dict(size=6, color="#1B3A6B"),
                name="Score",
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=9, color="#888"),
                    gridcolor="#DDE6F0",
                    linecolor="#DDE6F0",
                ),
                angularaxis=dict(
                    tickfont=dict(size=13, color="#1B3A6B", family="Arial"),
                    linecolor="#DDE6F0",
                    gridcolor="#DDE6F0",
                ),
                bgcolor="#F4F7FB",
            ),
            paper_bgcolor="#F4F7FB",
            showlegend=False,
            margin=dict(l=50, r=50, t=40, b=40),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cards:
        st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)

        dimensions = [
            ("📦 Delivery", scores["delivery"]["score"], scores["delivery"]["justification"]),
            ("🎯 Impact", scores["impact"]["score"], scores["impact"]["justification"]),
            ("⚠️ Risk Level", scores["risk_level"]["score"], scores["risk_level"]["justification"]),
            ("⚡ Efficiency", scores["efficiency"]["score"], scores["efficiency"]["justification"]),
        ]

        for label, score, justification in dimensions:
            if score >= 70:
                card_cls = "score-card score-card-green"
                score_color = "#2E7D32"
            elif score >= 40:
                card_cls = "score-card score-card-amber"
                score_color = "#E65100"
            else:
                card_cls = "score-card score-card-red"
                score_color = "#B71C1C"

            st.markdown(
                f"""
                <div class="{card_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="score-card-label">{label}</span>
                        <span class="score-card-score" style="color:{score_color};">{score}</span>
                    </div>
                    <div class="score-card-just">{justification}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# FEATURE 3 — ASK THE DOCUMENT
# =============================================================================
if st.session_state.document_text:
    st.markdown(
        '<div class="section-header">💬 Step 3 — Ask the Document</div>',
        unsafe_allow_html=True,
    )

    SUGGESTED_QUESTIONS = [
        "What were the main implementation challenges?",
        "Is this project on track?",
        "What outcomes were achieved?",
        "What does the funder require next?",
    ]

    st.markdown(
        "<div style='font-size:13px; font-weight:600; color:#1B3A6B; "
        "margin-bottom:8px;'>Quick questions:</div>",
        unsafe_allow_html=True,
    )
    q_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for idx, (col, question) in enumerate(zip(q_cols, SUGGESTED_QUESTIONS)):
        with col:
            if st.button(question, key=f"sq_{idx}", use_container_width=True):
                with st.spinner("🤖 Analysing document..."):
                    try:
                        ans = answer_question(
                            question,
                            st.session_state.document_text,
                            gemini_client,
                        )
                        st.session_state.chat_history.append(
                            {"role": "user", "content": question}
                        )
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": ans}
                        )
                    except Exception as exc:
                        st.error(f"Could not retrieve an answer. Please try again.\n\n`{exc}`")

    st.markdown("<br>", unsafe_allow_html=True)

    # Render chat history
    if st.session_state.chat_history:
        chat_html = "<div class='chat-wrap'>"
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += (
                    f"<div class='chat-user'>"
                    f"<div class='chat-label'>YOU</div>"
                    f"{msg['content']}</div>"
                )
            else:
                safe_content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
                chat_html += (
                    f"<div class='chat-ai'>"
                    f"<div class='chat-label'>🔬 DOCUSENSE</div>"
                    f"{safe_content}</div>"
                )
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # Chat input (sticks to bottom of page)
    user_question = st.chat_input(
        "Ask anything about this project document — e.g. 'What are the disbursement triggers?'"
    )
    if user_question:
        with st.spinner("🤖 Analysing document..."):
            try:
                ans = answer_question(
                    user_question,
                    st.session_state.document_text,
                    gemini_client,
                )
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_question}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ans}
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not retrieve an answer. Please try again.\n\n`{exc}`")

# =============================================================================
# FEATURE 4 — EXPORT REPORT
# =============================================================================
if st.session_state.extracted_data and st.session_state.scores:
    st.markdown(
        '<div class="section-header">📝 Step 4 — Generate & Export Stakeholder Report</div>',
        unsafe_allow_html=True,
    )

    col_gen, _ = st.columns([1, 3])
    with col_gen:
        gen_btn = st.button(
            "📄 Generate Stakeholder Report",
            type="primary",
            use_container_width=True,
        )

    if gen_btn:
        with st.spinner("✍️ Writing professional stakeholder report..."):
            try:
                st.session_state.report_text = generate_stakeholder_report(
                    st.session_state.extracted_data,
                    st.session_state.scores,
                    gemini_client,
                )
            except Exception as exc:
                st.error(
                    f"**Report generation failed.** Please try again.\n\n`{exc}`"
                )

    if st.session_state.report_text:
        st.markdown("#### Generated Stakeholder Report")
        safe_report = (
            st.session_state.report_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        st.markdown(
            f'<div class="report-box">{safe_report}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        project_name = (
            st.session_state.extracted_data.get("project_name", "Project")
            .replace(" ", "_")
            .replace("/", "-")
        )
        filename = f"{project_name}_DocuSense_Report.txt"

        st.download_button(
            label="⬇️ Download Report as .txt",
            data=st.session_state.report_text.encode("utf-8"),
            file_name=filename,
            mime="text/plain",
        )
