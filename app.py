from __future__ import annotations

import os
from copy import deepcopy
from html import escape

import streamlit as st
from dotenv import load_dotenv

import database
from pipeline import (
    PipelineError,
    PipelineResult,
    RubricDecomposition,
    generate_rubric_with_status,
    run_pipeline,
)

load_dotenv()
database.init_database()

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    # Local development: keys come from the .env file.
    pass

st.set_page_config(
    page_title="EduGrade",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400&family=Source+Serif+4:opsz,wght@8..60,400&display=swap');

:root {
  --paper: #f4f5f3;
  --surface: #ffffff;
  --ink: #1c2428;
  --muted: #5b676d;
  --faint: #8a949a;
  --rule: #d8dddf;
  --accent: #1d4b5c;
  --accent-soft: #e4edf0;
  --good: #2f6b4f;
  --mid: #8a6a1e;
  --low: #8c4a3f;
  --serif: 'Source Serif 4', 'Iowan Old Style', Georgia, serif;
  --sans: 'Public Sans', 'Helvetica Neue', Arial, sans-serif;
}

html, body, .stApp, [class*="css"] {
  font-family: var(--sans);
  color: var(--ink);
}
.stApp { background: var(--paper); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1180px; padding: 2.2rem 2rem 6rem; }

/* No bold anywhere: hierarchy comes from size, colour and space. */
*, b, strong, h1, h2, h3, h4, h5, h6, label, button, th, summary,
[data-testid="stMarkdownContainer"] strong {
  font-weight: 400 !important;
}

h1, h2, h3, h4 { font-family: var(--serif); color: var(--ink); letter-spacing: -0.01em; }
h3 { font-size: 1.45rem; line-height: 1.25; margin: 0 0 .25rem; padding: 0; }
p, li { line-height: 1.6; }

/* Masthead */
.mast { display: flex; align-items: baseline; justify-content: space-between;
        gap: 2rem; padding-bottom: 1.1rem; margin-bottom: .4rem;
        border-bottom: 1px solid var(--ink); }
.mast .name { font-family: var(--serif); font-size: 1.9rem; letter-spacing: -0.02em; }
.mast .line { color: var(--muted); font-size: .92rem; max-width: 34rem; text-align: right; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs button[data-baseweb="tab"] { padding: .9rem 0; height: auto; background: transparent;
  color: var(--muted); font-size: .98rem; }
.stTabs button[data-baseweb="tab"][aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem; }

/* Sections */
.section { margin: 2.6rem 0 1rem; padding-top: 1.2rem; border-top: 1px solid var(--rule); }
.section.first { border-top: 0; margin-top: 0; padding-top: 0; }
.section h3 { margin: 0; }
.section p { color: var(--muted); margin: .3rem 0 0; font-size: .93rem; max-width: 42rem; }

/* Inputs */
label, [data-testid="stWidgetLabel"] p { font-size: .86rem !important; color: var(--muted) !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div {
  background: var(--surface) !important; border: 1px solid var(--rule) !important;
  border-radius: 3px !important; color: var(--ink) !important; font-family: var(--sans) !important;
  box-shadow: none !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
  border-color: var(--accent) !important; outline: 2px solid var(--accent-soft) !important;
}
.stTextArea textarea { line-height: 1.55; }
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface); border: 1px dashed #b7c0c4; border-radius: 3px;
}
[data-testid="stFileUploaderDropzone"] button { border-radius: 3px; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 3px; padding: .55rem 1.2rem; font-size: .95rem;
  border: 1px solid var(--ink); background: transparent; color: var(--ink);
  transition: background .15s, color .15s;
}
.stButton > button:hover { background: var(--ink); color: #fff; border-color: var(--ink); }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.stButton > button[kind="primary"]:hover { background: #163b49; border-color: #163b49; }
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* Pipeline track */
.track { display: grid; grid-template-columns: repeat(6, 1fr); margin: 1.2rem 0 .6rem; }
.stop { position: relative; padding-top: 1.35rem; font-size: .85rem; color: var(--faint); }
.stop::before { content: ""; position: absolute; top: .42rem; left: 0; right: 0;
  height: 1px; background: var(--rule); }
.stop::after { content: ""; position: absolute; top: 0; left: 0; width: .85rem; height: .85rem;
  border-radius: 50%; background: var(--paper); border: 1px solid #aab4b9; box-sizing: border-box; }
.stop.done { color: var(--ink); }
.stop.done::before { background: var(--accent); }
.stop.done::after { background: var(--accent); border-color: var(--accent); }
.stop.active { color: var(--accent); }
.stop.active::after { border: 3px solid var(--accent); background: var(--paper); }
.stop.halt { color: var(--low); }
.stop.halt::after { background: var(--low); border-color: var(--low); }
.log { color: var(--muted); font-size: .88rem; line-height: 1.7; margin: .4rem 0 0; }
.log div::before { content: "-"; margin-right: .6rem; color: var(--faint); }

/* Notes (replace default coloured alert boxes) */
.note { border-left: 2px solid var(--accent); background: var(--surface);
  padding: .8rem 1.1rem; margin: .8rem 0; font-size: .93rem; color: var(--ink); }
.note.warn { border-color: var(--mid); }
.note.error { border-color: var(--low); }
.note.good { border-color: var(--good); }
.note .t { color: var(--muted); display: block; font-size: .84rem; margin-bottom: .15rem; }

/* Score block */
.scorehead { display: grid; grid-template-columns: minmax(180px, 260px) 1fr; gap: 3rem; align-items: start; }
.bigscore { font-family: var(--serif); font-size: 6.2rem; line-height: .95; letter-spacing: -0.04em; }
.bigscore small { font-size: 1.1rem; color: var(--faint); letter-spacing: 0; margin-left: .35rem; }
.level { color: var(--accent); font-size: 1.05rem; margin-top: .6rem; }
.summary { font-family: var(--serif); font-size: 1.22rem; line-height: 1.65; max-width: 40rem; margin: 0; }

/* Ledger */
.ledger { margin-top: .4rem; }
.ledger-row { display: grid; grid-template-columns: minmax(150px, 1.1fr) 2fr 86px;
  gap: 1.6rem; align-items: center; padding: 1.05rem 0; border-bottom: 1px solid var(--rule); }
.ledger-row:first-child { border-top: 1px solid var(--ink); }
.l-name { font-family: var(--serif); font-size: 1.08rem; }
.l-level { font-family: var(--sans); font-size: .82rem; color: var(--muted); margin-top: .15rem; }
.l-level i { display: inline-block; width: .5rem; height: .5rem; border-radius: 50%;
  margin-right: .45rem; background: var(--faint); }
.l-level.Strong i { background: var(--good); }
.l-level.Partial i { background: var(--mid); }
.l-level.Limited i, .l-level.Not i { background: var(--low); }
.l-bar { height: 3px; background: var(--rule); position: relative; }
.l-bar span { position: absolute; inset: 0 auto 0 0; background: var(--accent); }
.l-score { text-align: right; font-family: var(--serif); font-size: 1.35rem; }
.l-score small { color: var(--faint); font-size: .85rem; }
.l-detail { grid-column: 1 / -1; color: var(--muted); font-size: .92rem; max-width: 46rem; margin-top: -.5rem; }

/* Lists */
.plain { margin: .3rem 0 0; padding: 0; list-style: none; }
.plain li { padding: .65rem 0 .65rem 1rem; border-left: 1px solid var(--rule); margin-bottom: .35rem; }
.steps { margin: .3rem 0 0; padding-left: 1.3rem; }
.steps li { padding: .3rem 0 .3rem .3rem; }
.sub { font-family: var(--serif); font-size: 1.12rem; margin: 1.6rem 0 .3rem; }

/* Review rows */
.review-head { font-family: var(--serif); font-size: 1.12rem; margin-bottom: .1rem; }
.review-meta { color: var(--muted); font-size: .82rem; margin-bottom: .5rem; }
.review-text { color: var(--ink); font-size: .93rem; margin: 0 0 .35rem; }
.review-fix { color: var(--muted); font-size: .9rem; margin: 0; }
.total { font-family: var(--serif); font-size: 2.6rem; letter-spacing: -0.02em; }
.total small { font-size: 1rem; color: var(--faint); }

/* Archive */
.arch-head, .arch-row { display: grid; grid-template-columns: 56px 1.4fr 1.8fr 1.1fr 90px 1.2fr; gap: 1rem;
  padding: .75rem 0; align-items: baseline; font-size: .92rem; }
.arch-head { color: var(--muted); font-size: .82rem; border-bottom: 1px solid var(--ink); }
.arch-row { border-bottom: 1px solid var(--rule); }
.arch-row .n { font-family: var(--serif); }

[data-testid="stExpander"] { border: 0; border-top: 1px solid var(--rule); border-radius: 0; background: transparent; margin-top: 1.2rem; }
[data-testid="stExpander"] details { border: 0 !important; border-radius: 0 !important; }
.log { margin-bottom: 1.2rem; }
[data-testid="stExpander"] summary { padding-left: 0; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--rule); }
.empty { color: var(--muted); padding: 2.5rem 0; max-width: 34rem; }
.empty .h { font-family: var(--serif); font-size: 1.35rem; color: var(--ink); margin-bottom: .3rem; }

@media (max-width: 900px) {
  .block-container { padding: 1.4rem 1rem 4rem; }
  .mast { flex-direction: column; gap: .4rem; }
  .mast .line { text-align: left; }
  .track { grid-template-columns: repeat(3, 1fr); row-gap: 1rem; }
  .scorehead { grid-template-columns: 1fr; gap: 1.2rem; }
  .bigscore { font-size: 4.6rem; }
  .ledger-row { grid-template-columns: 1fr 70px; gap: .6rem 1rem; }
  .l-bar { grid-column: 1 / -1; order: 3; }
  .arch-head { display: none; }
  .arch-row { grid-template-columns: 44px 1fr 1fr; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

PIPE_STEPS = [
    ("intake", "Intake"),
    ("identify", "Rubric"),
    ("anomaly", "Integrity check"),
    ("analyze", "Scoring"),
    ("generate", "Feedback"),
    ("approval", "Teacher review"),
]

STATUS_LABELS = {
    "awaiting_teacher_review": "Awaiting review",
    "halted_for_review": "Halted for review",
    "approved": "Approved",
}


# ---------------------------------------------------------------- helpers
def init_state():
    defaults = {
        "result": None,
        "approved_report": None,
        "assessment_id": None,
        "events": [],
        "active_step": None,
        "pipeline_halted": False,
        "generated_rubric": None,
        "rubric_text": "",
        "rubric_editor": "",
        "rubric_notice": None,
        "selected_history": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def esc(value) -> str:
    return escape(str(value if value is not None else ""))


def fmt(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return esc(value)


def note(text: str, tone: str = "info", title: str = "") -> None:
    head = f'<span class="t">{esc(title)}</span>' if title else ""
    st.markdown(
        f'<div class="note {tone}">{head}{esc(text)}</div>', unsafe_allow_html=True
    )


def section(title: str, blurb: str = "", first: bool = False) -> None:
    sub = f"<p>{esc(blurb)}</p>" if blurb else ""
    cls = "section first" if first else "section"
    st.markdown(
        f'<div class="{cls}"><h3>{esc(title)}</h3>{sub}</div>', unsafe_allow_html=True
    )


def level_for(total: float) -> str:
    if total >= 90:
        return "Excellent"
    if total >= 75:
        return "Proficient"
    if total >= 60:
        return "Developing"
    return "Beginning"


def level_class(compliance: str) -> str:
    return compliance.split()[0] if compliance else "Partial"


def render_track() -> None:
    done = {e["step"] for e in st.session_state.events}
    active = st.session_state.active_step
    parts = ['<div class="track">']
    for step, label in PIPE_STEPS:
        cls = "stop"
        if st.session_state.pipeline_halted and step == "approval":
            cls += " halt"
        elif step == active:
            cls += " active"
        elif step in done:
            cls += " done"
        parts.append(f'<div class="{cls}">{label}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_log(container) -> None:
    lines = "".join(f"<div>{esc(e['message'])}</div>" for e in st.session_state.events)
    container.markdown(f'<div class="log">{lines}</div>', unsafe_allow_html=True)


def render_ledger(rows: list[dict], with_detail: bool = False) -> None:
    """rows: criterion, score, max_score, compliance, rationale (optional)."""
    html = ['<div class="ledger">']
    for row in rows:
        max_score = float(row["max_score"]) or 1.0
        pct = max(0.0, min(100.0, float(row["score"]) / max_score * 100))
        comp = row.get("compliance") or ""
        html.append(
            '<div class="ledger-row">'
            f'<div><div class="l-name">{esc(row["criterion"])}</div>'
            f'<div class="l-level {level_class(comp)}"><i></i>{esc(comp)}</div></div>'
            f'<div class="l-bar"><span style="width:{pct:.1f}%"></span></div>'
            f'<div class="l-score">{fmt(row["score"])}<small> / {fmt(max_score)}</small></div>'
            + (
                f'<div class="l-detail">{esc(row.get("rationale", ""))}</div>'
                if with_detail and row.get("rationale")
                else ""
            )
            + "</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def plain_list(items: list[str]) -> None:
    body = "".join(f"<li>{esc(i)}</li>" for i in items)
    st.markdown(f'<ul class="plain">{body}</ul>', unsafe_allow_html=True)


def max_by_name(report: dict) -> dict[str, float]:
    return {c["name"]: c["max_score"] for c in report["rubric"]["criteria"]}


def render_report(report: dict) -> None:
    feedback = report["feedback"]
    limits = max_by_name(report)
    rows = [
        {
            "criterion": s["criterion"],
            "score": s["score"],
            "max_score": limits.get(s["criterion"], 25),
            "compliance": s["compliance"],
            "rationale": s["rationale"],
        }
        for s in report["scores"]
    ]

    st.markdown(
        '<div class="scorehead"><div>'
        f'<div class="bigscore">{fmt(feedback["total_score"])}<small>out of 100</small></div>'
        f'<div class="level">{esc(feedback["overall_level"])}</div></div>'
        f'<p class="summary">{esc(feedback["summary"])}</p></div>',
        unsafe_allow_html=True,
    )

    section("Marks by criterion")
    render_ledger(rows, with_detail=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="sub">What worked</div>', unsafe_allow_html=True)
        plain_list(feedback["strengths"])
    with right:
        st.markdown('<div class="sub">Where to focus</div>', unsafe_allow_html=True)
        plain_list(feedback["growth_priorities"])

    st.markdown('<div class="sub">For the next attempt</div>', unsafe_allow_html=True)
    steps = "".join(f"<li>{esc(i)}</li>" for i in feedback["next_steps"])
    st.markdown(f'<ol class="steps">{steps}</ol>', unsafe_allow_html=True)

    evidence = [
        (s["criterion"], s.get("evidence_from_submission") or [])
        for s in report["scores"]
    ]
    if any(e for _, e in evidence):
        with st.expander("Passages the marks are based on"):
            for name, quotes in evidence:
                if quotes:
                    st.markdown(f'<div class="sub">{esc(name)}</div>', unsafe_allow_html=True)
                    plain_list(quotes)


def safe_text_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    try:
        return uploaded_file.getvalue().decode("utf-8").strip()
    except UnicodeDecodeError:
        return uploaded_file.getvalue().decode("latin-1", errors="replace").strip()


init_state()

# ---------------------------------------------------------------- keys
with st.sidebar:
    st.markdown('<div class="sub" style="margin-top:0">API keys</div>', unsafe_allow_html=True)
    st.caption("Leave blank to use the keys in your .env file or Streamlit secrets.")
    google_key = st.text_input(
        "Google API key", type="password", key="sidebar_google_key"
    )
    groq_key = st.text_input("Groq API key", type="password", key="sidebar_groq_key")

    has_google = bool(
        google_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )
    has_groq = bool(groq_key or os.getenv("GROQ_API_KEY"))
    st.caption(f"Google: {'available' if has_google else 'missing'}")
    st.caption(f"Groq: {'available' if has_groq else 'missing'}")

# ---------------------------------------------------------------- masthead
st.markdown(
    '<div class="mast"><div class="name">EduGrade</div>'
    '<div class="line">Rubric-based marking and written feedback. '
    "Nothing reaches a student until a teacher approves it.</div></div>",
    unsafe_allow_html=True,
)

grade_tab, report_tab, archive_tab = st.tabs(["Grade", "Student report", "Archive"])

# ================================================================ GRADE
with grade_tab:
    section(
        "Assignment",
        "Describe the task and agree the rubric. The generated rubric can be edited before marking.",
        first=True,
    )

    meta_l, meta_r = st.columns(2, gap="large")
    with meta_l:
        student_name = st.text_input("Student name", placeholder="Full name")
    with meta_r:
        assignment_title = st.text_input(
            "Assignment title", placeholder="Photosynthesis essay"
        )

    left, right = st.columns(2, gap="large")
    with left:
        assignment_prompt = st.text_area(
            "Assignment prompt",
            height=230,
            placeholder="Write a 700-word argumentative essay on whether social media improves civic participation.",
        )
        prompt_file = st.file_uploader(
            "Or upload the prompt (.txt or .md)", type=["txt", "md"], key="prompt_file"
        )
        uploaded_prompt_text = safe_text_file(prompt_file)

    effective_prompt = "\n\n".join(
        p for p in [assignment_prompt.strip(), uploaded_prompt_text] if p
    )

    with right:
        generate_clicked = st.button("Generate rubric from prompt")

        if generate_clicked:
            if not effective_prompt.strip():
                st.session_state.rubric_notice = (
                    "error",
                    "Add an assignment prompt first, then generate the rubric.",
                )
            else:
                with st.spinner("Writing the rubric"):
                    try:
                        generated, problem = generate_rubric_with_status(
                            assignment_prompt=effective_prompt,
                            teacher_rubric=st.session_state.rubric_text,
                            google_api_key=google_key,
                        )
                        st.session_state.generated_rubric = generated.model_dump()
                        text = "\n".join(
                            f"{c.name} ({c.max_score:g}/100): {c.description}"
                            for c in generated.criteria
                        )
                        st.session_state.rubric_text = text
                        st.session_state.rubric_editor = text
                        st.session_state.rubric_notice = (
                            (
                                "warn",
                                "The Google model could not be reached, so a standard "
                                f"template rubric was used. Reason: {problem}",
                            )
                            if problem
                            else None
                        )
                    except PipelineError as exc:
                        st.session_state.rubric_notice = ("error", str(exc))

        rubric_text = st.text_area(
            "Rubric",
            height=230,
            placeholder="Four criteria, each with a weight, totalling 100. Generate one from the prompt or write your own.",
            key="rubric_editor",
        )
        st.session_state.rubric_text = rubric_text

        notice = st.session_state.rubric_notice
        if notice:
            note(notice[1], notice[0])
        if st.session_state.generated_rubric:
            g = RubricDecomposition.model_validate(st.session_state.generated_rubric)
            st.caption(
                f"{g.assignment_type}, {len(g.criteria)} criteria, "
                f"{sum(c.max_score for c in g.criteria):g} marks"
            )

    section(
        "Student submission",
        "Paste the answer, or upload a photo of handwritten work and it will be transcribed first.",
    )
    text_col, image_col = st.columns([1.25, 0.75], gap="large")
    with text_col:
        typed_submission = st.text_area(
            "Typed response",
            height=250,
            placeholder="Paste the student's answer here.",
        )
    with image_col:
        image_file = st.file_uploader(
            "Image of the answer",
            type=["png", "jpg", "jpeg", "webp"],
            key="answer_image",
        )
        if image_file:
            st.image(image_file, use_container_width=True)

    section("Marking")
    track_slot = st.empty()
    log_slot = st.empty()
    with track_slot.container():
        render_track()

    run_clicked = st.button("Run assessment", type="primary")

    if run_clicked:
        problems = []
        if not student_name.strip():
            problems.append("Enter the student name.")
        if not assignment_title.strip():
            problems.append("Enter the assignment title.")
        if not effective_prompt.strip():
            problems.append("Add an assignment prompt.")
        if not rubric_text.strip():
            problems.append("Add a rubric, or generate one from the prompt.")
        if not typed_submission.strip() and not image_file:
            problems.append("Add the student's answer as text or an image.")

        if problems:
            note(" ".join(problems), "error", "Cannot start yet")
        else:
            st.session_state.events = []
            st.session_state.active_step = "intake"
            st.session_state.pipeline_halted = False
            st.session_state.result = None
            st.session_state.approved_report = None

            def event_callback(step: str, message: str):
                st.session_state.active_step = step
                st.session_state.events.append({"step": step, "message": message})
                with track_slot.container():
                    render_track()
                render_log(log_slot)

            image_bytes = image_file.getvalue() if image_file else None
            image_mime = image_file.type if image_file else None

            detected_type = (st.session_state.generated_rubric or {}).get(
                "assignment_type", "Essay"
            )

            try:
                student_id = database.add_student(student_name.strip())
                assignment_id = database.add_assignment(
                    title=assignment_title.strip(),
                    prompt=effective_prompt,
                    rubric_text=rubric_text.strip(),
                    assignment_type=detected_type,
                )
                with st.spinner("Marking in progress"):
                    result = run_pipeline(
                        assignment_prompt=effective_prompt,
                        rubric_text=rubric_text,
                        typed_submission=typed_submission,
                        student_id=student_id,
                        assignment_id=assignment_id,
                        image_bytes=image_bytes,
                        image_mime_type=image_mime,
                        google_api_key=google_key,
                        groq_api_key=groq_key,
                        event_callback=event_callback,
                    )
                st.session_state.assessment_id = database.save_assessment(
                    student_id=student_id,
                    assignment_id=assignment_id,
                    submission_content=result.submission_text,
                    pipeline_result=result,
                )
                st.session_state.result = result.model_dump()
                st.session_state.pipeline_halted = result.status == "halted_for_review"
                st.session_state.active_step = None
                with track_slot.container():
                    render_track()
            except PipelineError as exc:
                st.session_state.active_step = None
                note(str(exc), "error", "The assessment stopped")
            except Exception as exc:  # noqa: BLE001
                st.session_state.active_step = None
                note(f"{type(exc).__name__}: {exc}", "error", "Unexpected error")

    raw_result = st.session_state.result
    if raw_result:
        result = PipelineResult.model_validate(raw_result)

        section(
            "Teacher review",
            "Adjust any score or wording. The student sees only what you approve.",
        )

        if result.status == "halted_for_review":
            note(
                result.anomaly.teacher_note,
                "warn",
                f"Automatic scoring stopped. Severity: {result.anomaly.severity}.",
            )
            if result.anomaly.reasons:
                plain_list(result.anomaly.reasons)
            st.caption(
                "This is a routing decision, not a finding of plagiarism. "
                "The check has no external plagiarism database."
            )
        else:
            limits = {c.name: c.max_score for c in result.rubric.criteria}
            edited_scores = []

            for score in result.scores:
                max_score = float(limits[score.criterion])
                info, field = st.columns([1.6, 1], gap="large")
                with info:
                    st.markdown(
                        f'<div class="review-head">{esc(score.criterion)}</div>'
                        f'<div class="review-meta">{esc(score.compliance)}, out of {fmt(max_score)}</div>'
                        f'<p class="review-text">{esc(score.rationale)}</p>'
                        f'<p class="review-fix">Suggested improvement: {esc(score.improvement)}</p>',
                        unsafe_allow_html=True,
                    )
                with field:
                    edited = st.number_input(
                        "Score",
                        min_value=0.0,
                        max_value=max_score,
                        value=min(float(score.score), max_score),
                        step=0.5,
                        key=f"edit_score_{score.criterion}",
                    )
                edited_scores.append((score.criterion, edited))
                st.markdown(
                    '<div style="border-bottom:1px solid var(--rule);margin:.6rem 0 1rem"></div>',
                    unsafe_allow_html=True,
                )

            feedback = result.feedback
            if feedback:
                summary_edit = st.text_area(
                    "Overall feedback for the student",
                    value=feedback.summary,
                    height=130,
                    key="summary_edit",
                )
                next_steps_edit = st.text_area(
                    "Next steps, one per line",
                    value="\n".join(feedback.next_steps),
                    height=120,
                    key="steps_edit",
                )
                teacher_note = st.text_area(
                    "Private note for your records",
                    value=feedback.teacher_audit_note,
                    height=100,
                    key="audit_edit",
                )

                total_after_edits = round(sum(v for _, v in edited_scores), 2)
                st.markdown(
                    f'<div class="total">{fmt(total_after_edits)}<small> / 100 after your adjustments</small></div>',
                    unsafe_allow_html=True,
                )

                if st.button("Approve and publish report", type="primary"):
                    approved = deepcopy(result.model_dump())
                    score_map = dict(edited_scores)
                    for item in approved["scores"]:
                        item["score"] = score_map[item["criterion"]]

                    approved["feedback"]["total_score"] = total_after_edits
                    approved["feedback"]["summary"] = summary_edit.strip()
                    approved["feedback"]["next_steps"] = [
                        line.strip()
                        for line in next_steps_edit.splitlines()
                        if line.strip()
                    ]
                    approved["feedback"]["teacher_audit_note"] = teacher_note.strip()
                    approved["feedback"]["overall_level"] = level_for(total_after_edits)
                    approved["teacher_approved"] = True
                    st.session_state.approved_report = approved

                    database.approve_assessment(
                        assessment_id=st.session_state.assessment_id,
                        edited_scores=edited_scores,
                        total_score=total_after_edits,
                        overall_level=approved["feedback"]["overall_level"],
                        summary=summary_edit.strip(),
                        next_steps=approved["feedback"]["next_steps"],
                        teacher_audit_note=teacher_note.strip(),
                    )
                    note(
                        "The Student report tab now shows the approved version.",
                        "good",
                        "Report published",
                    )
                    try:
                        import vector_db

                        vector_db.add_assessment(
                            assessment_id=st.session_state.assessment_id,
                            assignment_prompt=effective_prompt,
                            assignment_type=result.rubric.assignment_type,
                            rubric=rubric_text,
                            submission=result.submission_text,
                            feedback=summary_edit.strip(),
                        )
                    except Exception as exc:  # noqa: BLE001
                        note(
                            f"The report was saved, but it could not be added to search. {exc}",
                            "warn",
                        )

        with st.expander("Full structured result"):
            st.json(result.model_dump())

# ================================================================ STUDENT REPORT
with report_tab:
    approved = st.session_state.approved_report
    draft = st.session_state.result

    report = None
    if approved:
        report = approved
        note("Approved by the teacher.", "good")
    elif draft and draft.get("status") == "awaiting_teacher_review":
        report = draft
        note(
            "This is a draft. The teacher has not approved it yet, so students should not see it.",
            "warn",
            "Preview",
        )
    elif draft and draft.get("status") == "halted_for_review":
        note(
            "This submission was sent to the teacher for manual review before any scoring.",
            "warn",
        )
    else:
        st.markdown(
            '<div class="empty"><div class="h">No report yet</div>'
            "Run an assessment on the Grade tab. The report appears here once it is ready.</div>",
            unsafe_allow_html=True,
        )

    if report and report.get("feedback"):
        render_report(report)

# ================================================================ ARCHIVE
with archive_tab:
    section("Previous assessments", first=True)

    history = database.get_assessment_history()

    if not history:
        st.markdown(
            '<div class="empty"><div class="h">Nothing archived yet</div>'
            "Completed assessments are stored here so you can find and review them later.</div>",
            unsafe_allow_html=True,
        )
    else:
        search_name = st.text_input(
            "Filter by student or assignment", placeholder="Name or title"
        )
        rows = history
        if search_name.strip():
            q = search_name.lower()
            rows = [
                i
                for i in history
                if q in i["student_name"].lower() or q in i["assignment_title"].lower()
            ]

        st.caption(f"{len(rows)} of {len(history)} assessments")
        st.markdown(
            '<div class="arch-head"><div>No.</div><div>Student</div><div>Assignment</div>'
            "<div>Status</div><div>Score</div><div>Date</div></div>",
            unsafe_allow_html=True,
        )
        for item in rows:
            score = (
                item["teacher_score"]
                if item["teacher_score"] is not None
                else item["ai_score"]
            )
            st.markdown(
                '<div class="arch-row">'
                f'<div class="n">{esc(item["assessment_id"])}</div>'
                f'<div>{esc(item["student_name"])}</div>'
                f'<div>{esc(item["assignment_title"])}</div>'
                f'<div>{esc(STATUS_LABELS.get(item["status"], str(item["status"]).replace("_", " ").capitalize()))}</div>'
                f'<div>{fmt(score) + " / 100" if score is not None else "Pending"}</div>'
                f'<div>{esc(str(item["created_at"])[:16].replace("T", " "))}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"view_{item['assessment_id']}"):
                st.session_state.selected_history = database.get_assessment_details(
                    item["assessment_id"]
                )

        selected = st.session_state.selected_history
        if selected:
            section(
                f"Assessment {selected['assessment_id']}",
                f"{selected['student_name']}, {selected['assignment_title']}",
            )
            with st.expander("Assignment prompt"):
                st.write(selected["prompt"])
            with st.expander("Student submission"):
                st.write(selected["submission_content"])

            if selected["scores"]:
                render_ledger(
                    [
                        {
                            "criterion": s["criterion_name"],
                            "score": s["score"],
                            "max_score": s["max_score"],
                            "compliance": s["compliance"],
                            "rationale": s["rationale"],
                        }
                        for s in selected["scores"]
                    ],
                    with_detail=True,
                )
            fb = selected["feedback"]
            if fb:
                st.markdown(
                    f'<div class="sub">{fmt(fb["total_score"])} / 100, {esc(fb["overall_level"])}</div>'
                    f'<p>{esc(fb["summary"])}</p>',
                    unsafe_allow_html=True,
                )

    section(
        "Search by meaning",
        "Find earlier assessments by topic, for example essays about remote work.",
    )
    semantic_query = st.text_input(
        "Describe what you are looking for", key="semantic_query"
    )
    if st.button("Search"):
        if not semantic_query.strip():
            note("Type a few words describing what you want to find.", "warn")
        else:
            try:
                import vector_db

                with st.spinner("Searching"):
                    matches = vector_db.search_similar(
                        query=semantic_query.strip(), n_results=5, max_distance=1.5
                    )
            except Exception as exc:  # noqa: BLE001
                matches = None
                note(f"Search is unavailable right now. {exc}", "error")

            if matches is not None:
                if not matches:
                    note("No close matches. Try different words.")
                for match in matches:
                    meta = match["metadata"]
                    with st.expander(
                        f"Assessment {meta.get('assessment_id', '?')}, "
                        f"{meta.get('assignment_type', 'Unknown')}"
                    ):
                        st.caption(f"Distance {match['distance']:.3f}. Lower is closer.")
                        st.text(match["document"])
