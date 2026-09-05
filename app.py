from __future__ import annotations
import database
database.init_database()
import json
import os
from copy import deepcopy

import database
database.init_database()
import vector_db

import streamlit as st
from dotenv import load_dotenv

from pipeline import PipelineError, PipelineResult, RubricDecomposition, generate_rubric, run_pipeline

load_dotenv()

st.set_page_config(
    page_title="EduGrade AI  Assessment & Feedback Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --ink: #172033;
  --muted: #5d6b82;
  --blue: #355f8a;
  --blue-2: #4f7ea8;
  --slate: #eef3f7;
  --warm: #fbf8f3;
  --line: #dce5ec;
  --good: #2f7d5a;
  --warn: #a66a1f;
  --bad: #a24545;
}
.stApp {
  background:
    radial-gradient(circle at 8% 0%, rgba(79,126,168,.10), transparent 26%),
    linear-gradient(180deg, #fbfcfd 0%, #f6f8fa 100%);
}
.block-container {padding-top: 1.5rem; max-width: 1450px;}
h1, h2, h3 {color: var(--ink); letter-spacing: -0.02em;}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #172033 0%, #223149 100%);
}
[data-testid="stSidebar"] * {color: #f4f7fa;}
[data-testid="stSidebar"] input {
  color: #172033 !important;
  background: #f9fbfc !important;
}
.hero {
  background: linear-gradient(135deg, #243b55, #355f8a);
  color: white;
  border-radius: 22px;
  padding: 26px 30px;
  box-shadow: 0 14px 38px rgba(31, 52, 76, .18);
  margin-bottom: 18px;
}
.hero h1 {color: white; margin: 0 0 4px 0; font-size: 2rem;}
.hero p {color: #dce9f5; margin: 0; max-width: 900px;}
.agent-card, .score-card, .alert-card, .report-card {
  background: rgba(255,255,255,.94);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 7px 22px rgba(37,56,78,.06);
}
.pipeline {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 8px;
  margin: 12px 0 20px 0;
}
.pipe-step {
  border-radius: 14px;
  border: 1px solid #dce5ec;
  background: white;
  padding: 12px 10px;
  text-align: center;
  font-size: .84rem;
  color: #526177;
}
.pipe-step.done {
  background: #edf7f2;
  border-color: #bcdcca;
  color: #286447;
  font-weight: 700;
}
.pipe-step.active {
  background: #eaf2fa;
  border-color: #a9c5dd;
  color: #294f72;
  font-weight: 700;
}
.pipe-step.halt {
  background: #fff2ef;
  border-color: #edc0ba;
  color: #8e3f37;
  font-weight: 700;
}
.badge {
  display:inline-block; padding:5px 10px; border-radius:999px;
  font-weight:700; font-size:.78rem;
}
.badge-strong {background:#e5f4eb; color:#286447;}
.badge-partial {background:#fff2d9; color:#855a18;}
.badge-limited {background:#fde9d8; color:#95531e;}
.badge-none {background:#f8dfdf; color:#923d3d;}
.score-big {font-size: 2.5rem; font-weight: 800; color:#294f72; line-height:1;}
.muted {color:#69778d;}
.smallcaps {font-size:.75rem; letter-spacing:.08em; text-transform:uppercase; font-weight:800; color:#65748a;}
div[data-testid="stMetric"] {
  background:white; border:1px solid var(--line); border-radius:16px; padding:12px 14px;
}
.stButton > button {
  border-radius: 12px;
  font-weight: 700;
}
@media (max-width: 900px) {
  .pipeline {grid-template-columns: repeat(2, 1fr);}
}
</style>
""",
    unsafe_allow_html=True,
)

PIPE_STEPS = [
    ("intake", "1  Intake"),
    ("identify", "2  Decompose"),
    ("anomaly", "3  Anomaly"),
    ("analyze", "4  Parallel Score"),
    ("generate", "5  Feedback"),
    ("approval", "6  Approval"),
]


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_pipeline():
    done = {e["step"] for e in st.session_state.events}
    active = st.session_state.active_step
    html = ['<div class="pipeline">']
    for step, label in PIPE_STEPS:
        cls = "pipe-step"
        if st.session_state.pipeline_halted and step == "approval":
            cls += " halt"
        elif step == active:
            cls += " active"
        elif step in done:
            cls += " done"
        html.append(f'<div class="{cls}">{label}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def compliance_badge(value: str) -> str:
    css = {
        "Strong": "badge-strong",
        "Partial": "badge-partial",
        "Limited": "badge-limited",
        "Not Demonstrated": "badge-none",
    }.get(value, "badge-partial")
    return f'<span class="badge {css}">{value}</span>'


def safe_text_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    try:
        return uploaded_file.getvalue().decode("utf-8").strip()
    except UnicodeDecodeError:
        return uploaded_file.getvalue().decode("latin-1", errors="replace").strip()


init_state()

with st.sidebar:

    st.markdown("## 🎓 EduGrade AI")

    st.caption("AI-Powered Assessment Platform")

    st.divider()

    st.markdown("### ⚙️ Settings")

    with st.expander("API Configuration", expanded=False):

        google_key = st.text_input(
            "Google API Key",
            type="password",
            value="",
            placeholder="Uses .env automatically if empty",
            key="sidebar_google_key",
        )

        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value="",
            placeholder="Uses .env automatically if empty",
            key="sidebar_groq_key",
        )

        has_google = bool(
            google_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        has_groq = bool(
            groq_key
            or os.getenv("GROQ_API_KEY")
        )

        if has_google:
            st.success("Google AI connected")
        else:
            st.warning("Google AI key missing")

        if has_groq:
            st.success("Groq AI connected")
        else:
            st.warning("Groq AI key missing")

    st.divider()

    st.caption(
        "Teacher-controlled AI assessment and feedback."
    )
st.markdown(
    """
<div class="hero">
  <div class="smallcaps" style="color:#bcd4e8">Assessment & Feedback Agent</div>
  <h1>Explainable grading, with a teacher in control.</h1>
  <p>Multimodal intake, structured rubric decomposition, anomaly routing,
  parallel criterion scoring, actionable feedback, and an auditable approval queue.</p>
</div>
""",
    unsafe_allow_html=True,
)

teacher_tab, student_tab, history_tab = st.tabs(
    [
        " Teacher Console",
        " Student Report",
        " Previous Submissions"
    ]
)

with teacher_tab:
    st.subheader("Assignment setup")
    left, right = st.columns([1, 1])
    student_name = st.text_input(
        "Student name",
        placeholder="Enter student name",
    )

    assignment_title = st.text_input(
        "Assignment title",
        placeholder="e.g. Photosynthesis Essay",
    )

    with left:
        assignment_prompt = st.text_area(
            "Essay / assignment prompt",
            height=150,
            placeholder="Example: Write a 700-word argumentative essay on whether social media improves civic participation...",
        )
        prompt_file = st.file_uploader(
            "Or upload the prompt (.txt / .md)",
            type=["txt", "md"],
            key="prompt_file",
        )
        uploaded_prompt_text = safe_text_file(prompt_file)

    effective_prompt = "\n\n".join(
        p for p in [assignment_prompt.strip(), uploaded_prompt_text] if p
    )

    with right:
        st.markdown("### AI-generated rubric")
        st.caption("Generate a rubric from the assignment prompt. You can edit it before assessment.")

        generate_clicked = st.button(
            "? Generate rubric with AI",
            use_container_width=True,
        )

        if generate_clicked:
            if not effective_prompt.strip():
                st.error("Enter or upload an assignment prompt first.")
            else:
                st.info("Starting rubric generation...")
                with st.spinner("Generating rubric..."):
                    try:
                        st.info("Calling rubric generator...")
                        generated = generate_rubric(
                            assignment_prompt=effective_prompt,
                            teacher_rubric=st.session_state.rubric_text,
                            google_api_key=google_key,
                        )
                        st.session_state.generated_rubric = generated.model_dump()

                        generated_text = "\n".join(
                            f"{c.name} ({c.max_score:g}/100): {c.description}"
                            for c in generated.criteria
                        )

                        st.session_state.rubric_text = generated_text
                        st.session_state.rubric_editor = generated_text
                    except PipelineError as exc:
                        st.error(str(exc))

        rubric_text = st.text_area(
            "Rubric - editable by teacher",
            value=st.session_state.rubric_text,
            height=220,
            placeholder="Click Generate rubric with AI after entering the assignment prompt.",
            key="rubric_editor",
        )
        st.session_state.rubric_text = rubric_text

        if st.session_state.generated_rubric:
            generated = RubricDecomposition.model_validate(st.session_state.generated_rubric)
            st.caption(
                f"Detected type: {generated.assignment_type}  "
                f"{len(generated.criteria)} criteria  "
                f"{sum(c.max_score for c in generated.criteria):g}/100"
            )

    st.subheader("Student submission")
    text_col, image_col = st.columns([1.25, 0.75])
    with text_col:
        typed_submission = st.text_area(
            "Paste student response",
            height=250,
            placeholder="Paste the student's answer here, or upload a handwritten/printed image.",
        )
    with image_col:
        image_file = st.file_uploader(
            "Upload answer image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Gemini multimodal intake transcribes the image before grading.",
            key="answer_image",
        )
        if image_file:
            st.image(image_file, caption="Submission image", use_container_width=True)


    st.markdown("### Live agent pipeline")
    pipeline_placeholder = st.empty()
    log_placeholder = st.empty()

    with pipeline_placeholder.container():
        render_pipeline()

    run_clicked = st.button(
        " Run multi-agent assessment",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        st.session_state.events = []
        st.session_state.active_step = "intake"
        st.session_state.pipeline_halted = False
        st.session_state.result = None
        st.session_state.approved_report = None

        def event_callback(step: str, message: str):
            st.session_state.active_step = step
            st.session_state.events.append({"step": step, "message": message})
            with pipeline_placeholder.container():
                render_pipeline()
            with log_placeholder.container():
                for event in st.session_state.events:
                    st.markdown(f"**{event['step'].title()}**  {event['message']}")

        image_bytes = image_file.getvalue() if image_file else None
        image_mime = image_file.type if image_file else None
        if not student_name.strip():
            st.error("Please enter the student name.")
            st.stop()

        if not assignment_title.strip():
            st.error("Please enter the assignment title.")
            st.stop()

        student_id = database.add_student(student_name.strip())

        assignment_id = database.add_assignment(
            title=assignment_title.strip(),
            prompt=effective_prompt,
            rubric=rubric_text.strip(),
            assignment_type="Essay",
        )
        try:
            with st.spinner("Agents are working..."):
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
            assessment_id = database.save_assessment(
                student_id=student_id,
                assignment_id=assignment_id,
                submission_content=result.submission_text,
                pipeline_result=result,

      )       

            st.session_state.result = result.model_dump()
            st.session_state.assessment_id = assessment_id
            st.session_state.pipeline_halted = result.status == "halted_for_review"
            st.session_state.active_step = None
            with pipeline_placeholder.container():
                render_pipeline()
        except PipelineError as exc:
            st.session_state.active_step = None
            st.error(str(exc))
        except Exception as exc:
            st.session_state.active_step = None
            st.error(f"Unexpected application error: {exc}")

    raw_result = st.session_state.result
    if raw_result:
        result = PipelineResult.model_validate(raw_result)

        st.divider()
        st.subheader("Teacher audit queue")

        if result.status == "halted_for_review":
            st.markdown(
                f"""
<div class="alert-card">
  <div class="smallcaps">Manual review required</div>
  <h3 style="margin:.3rem 0">Automated scoring was halted</h3>
  <p><b>Severity:</b> {result.anomaly.severity.title()}</p>
  <p>{result.anomaly.teacher_note}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            if result.anomaly.reasons:
                st.write("**Why it was routed:**")
                for reason in result.anomaly.reasons:
                    st.write(f" {reason}")
            st.info(
                "This is intentionally a routing decision, not a plagiarism verdict. "
                "The anomaly model has no external plagiarism corpus."
            )
        else:
            st.success("Draft grading complete. Teacher approval is required before publication.")

            st.markdown("#### Rubric decomposition")
            for criterion in result.rubric.criteria:
                with st.expander(f"{criterion.name}  {criterion.max_score} points"):
                    st.write(criterion.description)
                    for indicator in criterion.success_indicators:
                        st.write(f" {indicator}")

            st.markdown("#### Agent-generated scores  editable by teacher")
            edited_scores = []
            for score in result.scores:
                with st.container(border=True):
                    c1, c2 = st.columns([0.25, 0.75])
                    with c1:
                        edited = st.number_input(
                            f"{score.criterion} score",
                            min_value=0.0,
                            max_value=float(next(c.max_score for c in result.rubric.criteria if c.name == score.criterion)),
                            value=float(score.score),
                            step=0.5,
                            key=f"edit_score_{score.criterion}",
                        )
                    with c2:
                        st.markdown(compliance_badge(score.compliance), unsafe_allow_html=True)
                        st.write(score.rationale)
                        st.caption(f"Agent improvement: {score.improvement}")
                    edited_scores.append((score.criterion, edited))

            feedback = result.feedback
            if feedback:
                summary_edit = st.text_area(
                    "Overall feedback  teacher editable",
                    value=feedback.summary,
                    height=130,
                    key="summary_edit",
                )
                next_steps_edit = st.text_area(
                    "Next steps  one item per line",
                    value="\n".join(feedback.next_steps),
                    height=120,
                    key="steps_edit",
                )
                teacher_note = st.text_area(
                    "Private teacher audit note",
                    value=feedback.teacher_audit_note,
                    height=100,
                    key="audit_edit",
                )

                total_after_edits = round(sum(v for _, v in edited_scores), 2)
                st.metric("Teacher-adjusted total", f"{total_after_edits:.1f} / 100")

                if st.button(" Approve and publish student report", type="primary"):
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
                    approved["feedback"]["overall_level"] = (
                        "Excellent"
                        if total_after_edits >= 90
                        else "Proficient"
                        if total_after_edits >= 75
                        else "Developing"
                        if total_after_edits >= 60
                        else "Beginning"
                    )
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
                    vector_db.add_assessment(
                        assessment_id=st.session_state.assessment_id,
                        assignment_prompt=effective_prompt,
                        assignment_type=result.rubric.assignment_type,
                        rubric=rubric_text,
                        submission=result.submission_text,
                        feedback=summary_edit.strip(),
                    )
                    st.success("Published. The Student Report tab now shows the approved version.")

            with st.expander("Audit trail / raw structured result"):
                st.json(result.model_dump())

with student_tab:
    approved = st.session_state.approved_report
    draft = st.session_state.result

    if approved:
        report = approved
        st.success("Teacher-approved report")
    elif draft and draft.get("status") == "awaiting_teacher_review":
        report = draft
        st.warning("Preview only  this draft has not yet been approved by the teacher.")
    elif draft and draft.get("status") == "halted_for_review":
        report = None
        st.warning("This submission has been routed to the teacher for manual review.")
    else:
        report = None
        st.info("Run an assessment in the Teacher Console to generate a student report.")

    if report and report.get("feedback"):
        feedback = report["feedback"]
        scores = report["scores"]

        top1, top2, top3 = st.columns([1.15, 1, 1])
        with top1:
            st.markdown(
                f"""
<div class="score-card">
  <div class="smallcaps">Overall score</div>
  <div class="score-big">{feedback['total_score']:.1f}<span style="font-size:1rem;color:#738197"> / 100</span></div>
  <div style="margin-top:8px"><b>{feedback['overall_level']}</b></div>
</div>
""",
                unsafe_allow_html=True,
            )
        with top2:
            strongest = max(scores, key=lambda x: x["score"])
            st.metric(
                "Strongest criterion",
                strongest["criterion"],
                f"{strongest['score']:.1f}/{next(c.max_score for c in result.rubric.criteria if c.name == strongest['criterion']):g}",
            )
        with top3:
            growth = min(scores, key=lambda x: x["score"])
            st.metric(
                "Growth focus",
                growth["criterion"],
                f"{growth['score']:.1f}/{next(c.max_score for c in result.rubric.criteria if c.name == growth['criterion']):g}",
            )

        st.markdown("### Criterion breakdown")
        cols = st.columns(4)
        for col, score in zip(cols, scores):
            with col:
                st.markdown(
                    f"""
<div class="score-card">
  <div class="smallcaps">{score['criterion']}</div>
  <div style="font-size:1.9rem;font-weight:800;color:#294f72;margin:.25rem 0">
    {score['score']:.1f}<span style="font-size:.85rem;color:#738197"> / 25</span>
  </div>
  {compliance_badge(score['compliance'])}
  <p style="font-size:.9rem;margin-top:12px">{score['improvement']}</p>
</div>
""",
                    unsafe_allow_html=True,
                )

        st.markdown("### Your feedback")
        st.markdown(
            f"""
<div class="report-card">
  <h3 style="margin-top:0">Overall</h3>
  <p>{feedback['summary']}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        a, b = st.columns(2)
        with a:
            st.markdown("#### What you did well")
            for item in feedback["strengths"]:
                st.success(item)
        with b:
            st.markdown("#### Highest-impact growth areas")
            for item in feedback["growth_priorities"]:
                st.warning(item)

        st.markdown("#### Next attempt: action plan")
        for index, item in enumerate(feedback["next_steps"], start=1):
            st.write(f"**{index}.** {item}")

        with st.expander("Why did I receive these scores?"):
            for score in scores:
                st.markdown(f"**{score['criterion']}  {score['score']:.1f}/25**")
                st.write(score["rationale"])
                if score.get("evidence_from_submission"):
                    st.caption("Evidence considered: " + "  ".join(score["evidence_from_submission"]))


with history_tab:

    st.header("Previous Submissions")
    st.divider()

st.subheader("AI Semantic Search")

st.caption(
    "Search previous assessments using natural language."
)

semantic_query = st.text_input(
    "What are you looking for?",
    placeholder="Example: Find essays about remote work and productivity",
    key="semantic_query"
)

semantic_search_clicked = st.button(
    "Search Similar Assessments",
    type="primary",
    use_container_width=True
)

if semantic_search_clicked:

    if not semantic_query.strip():

        st.warning("Please enter something to search.")

    else:

        with st.spinner("Searching similar assessments..."):

            matches = vector_db.search_similar(
                query=semantic_query.strip(),
                n_results=5,
                max_distance=1.5,
            )

        if not matches:

            st.info(
                "No sufficiently similar assessments found."
            )

        else:

            st.success(
                f"Found {len(matches)} similar assessment(s)"
            )

            for index, match in enumerate(matches, start=1):

                metadata = match["metadata"]
                distance = match["distance"]
                document = match["document"]

                assessment_id = metadata.get(
                    "assessment_id",
                    "Unknown"
                )

                assignment_type = metadata.get(
                    "assignment_type",
                    "Unknown"
                )

                with st.expander(
                    f"Result {index} — Assessment #{assessment_id}"
                ):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Assignment Type",
                            assignment_type
                        )

                    with col2:
                        st.metric(
                            "Similarity Distance",
                            f"{distance:.4f}"
                        )

                    st.markdown("### Assessment Content")

                    st.text(document)

    history = database.get_assessment_history()

    if not history:
        st.info("No previous assessments found.")

    else:

        st.caption(
            f"{len(history)} assessment(s) found in the database."
        )

        search_name = st.text_input(
            "Search by student or assignment",
            placeholder="Enter student name or assignment title"
        )

        filtered_history = history

        if search_name.strip():

            query = search_name.lower()

            filtered_history = [
                item
                for item in history
                if query in item["student_name"].lower()
                or query in item["assignment_title"].lower()
            ]

        for item in filtered_history:

            title = (
                f"#{item['assessment_id']} — "
                f"{item['student_name']} | "
                f"{item['assignment_title']}"
            )

            with st.expander(title):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Status",
                        item["status"]
                    )

                with col2:
                    score = (
                        item["teacher_score"]
                        or item["ai_score"]
                    )

                    if score is not None:
                        st.metric(
                            "Score",
                            f"{score:.1f}/100"
                        )
                    else:
                        st.metric(
                            "Score",
                            "Pending"
                        )

                with col3:
                    st.write(
                        "Created:",
                        item["created_at"]
                    )

                if st.button(
                    "View Full Assessment",
                    key=f"view_{item['assessment_id']}"
                ):

                    details = database.get_assessment_details(
                        item["assessment_id"]
                    )

                    st.session_state.selected_history = details


        selected = st.session_state.get(
            "selected_history"
        )

        if selected:

            st.divider()

            st.subheader(
                f"Assessment #{selected['assessment_id']}"
            )

            st.write(
                f"### Student: {selected['student_name']}"
            )

            st.write(
                f"### Assignment: {selected['assignment_title']}"
            )

            st.write("### Assignment Prompt")

            st.info(selected["prompt"])

            st.write("### Student Submission")

            st.text_area(
                "Submission",
                value=selected["submission_content"],
                height=250,
                disabled=True,
                key=f"history_submission_{selected['assessment_id']}"
            )

            st.write("### Criterion Scores")

            for score in selected["scores"]:

                with st.expander(
                    f"{score['criterion_name']} — "
                    f"{score['score']}/{score['max_score']}"
                ):

                    st.write(
                        "**Compliance:**",
                        score["compliance"]
                    )

                    st.write(
                        "**Rationale:**",
                        score["rationale"]
                    )

                    st.write(
                        "**Improvement:**",
                        score["improvement"]
                    )

            if selected["feedback"]:

                feedback = selected["feedback"]

                st.write("### Feedback")

                st.success(
                    f"{feedback['total_score']}/100 — "
                    f"{feedback['overall_level']}"
                )

                st.write(feedback["summary"])


