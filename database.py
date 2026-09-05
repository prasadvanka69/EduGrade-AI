import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone


DB_PATH = Path(__file__).parent / "edugrade.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database():
    conn = get_connection()

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        assignment_type TEXT NOT NULL,
        prompt TEXT NOT NULL,
        rubric_text TEXT,
        rubric_json TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        assignment_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        source TEXT DEFAULT 'text',
        filename TEXT,
        submitted_at TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(id),
        FOREIGN KEY (assignment_id) REFERENCES assignments(id)
    );

    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        submission_id INTEGER NOT NULL UNIQUE,
        status TEXT NOT NULL,
        ai_score REAL,
        teacher_score REAL,
        anomaly_json TEXT,
        models_json TEXT,
        created_at TEXT NOT NULL,
        approved_at TEXT,
        approved_by TEXT,
        FOREIGN KEY (submission_id) REFERENCES submissions(id)
    );

    CREATE TABLE IF NOT EXISTS criterion_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL,
        criterion_name TEXT NOT NULL,
        max_score REAL NOT NULL,
        score REAL NOT NULL,
        compliance TEXT,
        evidence TEXT,
        rationale TEXT,
        improvement TEXT,
        FOREIGN KEY (assessment_id) REFERENCES assessments(id)
    );

    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL UNIQUE,
        total_score REAL NOT NULL,
        overall_level TEXT,
        summary TEXT,
        strengths_json TEXT,
        growth_priorities_json TEXT,
        next_steps_json TEXT,
        teacher_note TEXT,
        FOREIGN KEY (assessment_id) REFERENCES assessments(id)
    );

    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER,
        action TEXT NOT NULL,
        details TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (assessment_id) REFERENCES assessments(id)
    );
    """)

    conn.commit()
    conn.close()


def now():
    return datetime.now(timezone.utc).isoformat()


def add_student(name, email=None):
    conn = get_connection()

    cursor = conn.execute(
        """
        INSERT INTO students (name, email, created_at)
        VALUES (?, ?, ?)
        """,
        (name, email, now())
    )

    conn.commit()
    student_id = cursor.lastrowid
    conn.close()

    return student_id


def add_assignment(title, assignment_type, prompt,
                   rubric_text="", rubric=None):
    conn = get_connection()

    rubric_json = json.dumps(rubric) if rubric is not None else None

    cursor = conn.execute(
        """
        INSERT INTO assignments
        (title, assignment_type, prompt, rubric_text, rubric_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            title,
            assignment_type,
            prompt,
            rubric_text,
            rubric_json,
            now()
        )
    )

    conn.commit()
    assignment_id = cursor.lastrowid
    conn.close()

    return assignment_id


def add_submission(student_id, assignment_id, content,
                   source="text", filename=None):
    conn = get_connection()

    cursor = conn.execute(
        """
        INSERT INTO submissions
        (student_id, assignment_id, content, source, filename, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            student_id,
            assignment_id,
            content,
            source,
            filename,
            now()
        )
    )

    conn.commit()
    submission_id = cursor.lastrowid
    conn.close()

    return submission_id


def add_assessment(submission_id, status="processing",
                   ai_score=None, anomaly=None, models=None):
    conn = get_connection()

    cursor = conn.execute(
        """
        INSERT INTO assessments
        (submission_id, status, ai_score, anomaly_json, models_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            submission_id,
            status,
            ai_score,
            json.dumps(anomaly) if anomaly is not None else None,
            json.dumps(models) if models is not None else None,
            now()
        )
    )

    conn.commit()
    assessment_id = cursor.lastrowid
    conn.close()

    return assessment_id


def add_criterion_score(
    assessment_id,
    criterion_name,
    max_score,
    score,
    compliance,
    evidence,
    rationale,
    improvement
):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO criterion_scores
        (
            assessment_id,
            criterion_name,
            max_score,
            score,
            compliance,
            evidence,
            rationale,
            improvement
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            assessment_id,
            criterion_name,
            max_score,
            score,
            compliance,
            json.dumps(evidence) if isinstance(evidence, list) else evidence,
            rationale,
            improvement
        )
    )

    conn.commit()
    conn.close()


def add_feedback(
    assessment_id,
    total_score,
    overall_level,
    summary,
    strengths=None,
    growth_priorities=None,
    next_steps=None,
    teacher_note=None
):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO feedback
        (
            assessment_id,
            total_score,
            overall_level,
            summary,
            strengths_json,
            growth_priorities_json,
            next_steps_json,
            teacher_note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            assessment_id,
            total_score,
            overall_level,
            summary,
            json.dumps(strengths or []),
            json.dumps(growth_priorities or []),
            json.dumps(next_steps or []),
            teacher_note
        )
    )

    conn.commit()
    conn.close()


def update_assessment_status(
    assessment_id,
    status,
    teacher_score=None,
    approved_by=None
):
    conn = get_connection()

    approved_at = now() if status == "approved" else None

    conn.execute(
        """
        UPDATE assessments
        SET
            status = ?,
            teacher_score = COALESCE(?, teacher_score),
            approved_at = COALESCE(?, approved_at),
            approved_by = COALESCE(?, approved_by)
        WHERE id = ?
        """,
        (
            status,
            teacher_score,
            approved_at,
            approved_by,
            assessment_id
        )
    )

    conn.commit()
    conn.close()


def add_audit_log(assessment_id, action, details=""):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO audit_log
        (assessment_id, action, details, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            assessment_id,
            action,
            details,
            now()
        )
    )

    conn.commit()
    conn.close()


def get_assessment_history():
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            a.id AS assessment_id,
            s.name AS student_name,
            ass.title AS assignment_title,
            ass.assignment_type,
            a.status,
            a.ai_score,
            a.teacher_score,
            a.created_at,
            a.approved_at
        FROM assessments a
        JOIN submissions sub ON a.submission_id = sub.id
        JOIN students s ON sub.student_id = s.id
        JOIN assignments ass ON sub.assignment_id = ass.id
        ORDER BY a.created_at DESC
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]

def save_assessment(
    student_id,
    assignment_id,
    submission_content,
    pipeline_result
):
    """
    Save one complete pipeline result into SQLite.
    """

    submission_id = add_submission(
        student_id=student_id,
        assignment_id=assignment_id,
        content=submission_content
    )

    assessment_id = add_assessment(
        submission_id=submission_id,
        status=pipeline_result.status,
        ai_score=(
            pipeline_result.feedback.total_score
            if pipeline_result.feedback
            else None
        ),
        anomaly=pipeline_result.anomaly.model_dump(),
        models=pipeline_result.models_used
    )

    for score in pipeline_result.scores:
        add_criterion_score(
            assessment_id=assessment_id,
            criterion_name=score.criterion,
            max_score=next(
                criterion.max_score
                for criterion in pipeline_result.rubric.criteria
                    if criterion.name == score.criterion
            ),
            score=score.score,
            compliance=score.compliance,
            evidence=score.evidence_from_submission,
            rationale=score.rationale,
            improvement=score.improvement,
        )

    if pipeline_result.feedback:
        add_feedback(
            assessment_id=assessment_id,
            total_score=pipeline_result.feedback.total_score,
            overall_level=pipeline_result.feedback.overall_level,
            summary=pipeline_result.feedback.summary,
            strengths=pipeline_result.feedback.strengths,
            growth_priorities=pipeline_result.feedback.growth_priorities,
            next_steps=pipeline_result.feedback.next_steps
        )

    add_audit_log(
        assessment_id=assessment_id,
        action="assessment_created",
        details=f"Pipeline status: {pipeline_result.status}"
    )

    return assessment_id
def approve_assessment(
    assessment_id,
    edited_scores,
    total_score,
    overall_level,
    summary,
    next_steps,
    teacher_audit_note,
):
    """Update an assessment with the teacher-approved final result."""

    with get_connection() as conn:

        conn.execute(
            """
            UPDATE assessments
            SET status = ?,
                ai_score = ?
            WHERE id = ?
            """,
            ("approved", total_score, assessment_id),
        )

        for criterion_name, score in edited_scores:
            conn.execute(
                """
                UPDATE criterion_scores
                SET score = ?
                WHERE assessment_id = ?
                  AND criterion_name = ?
                """,
                (score, assessment_id, criterion_name),
            )

        conn.execute(
            """
            UPDATE feedback
            SET total_score = ?,
                overall_level = ?,
                summary = ?,
                next_steps_json = ?,
                teacher_note = ?
            WHERE assessment_id = ?
            """,
            (
                total_score,
                overall_level,
                summary,
                json.dumps(next_steps),
                teacher_audit_note,
                assessment_id,
            ),
        )

        conn.execute(
            """
            INSERT INTO audit_log
            (assessment_id, action, details, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                assessment_id,
                "teacher_approved",
                f"Teacher approved final score: {total_score}/100",
                now(),
            ),
        )

        conn.commit()

def get_assessment_details(assessment_id):
    conn = get_connection()

    assessment = conn.execute(
        """
        SELECT
            a.id AS assessment_id,
            a.status,
            a.ai_score,
            a.teacher_score,
            a.created_at,
            a.approved_at,

            s.name AS student_name,
            sub.content AS submission_content,

            ass.title AS assignment_title,
            ass.assignment_type,
            ass.prompt,
            ass.rubric_text

        FROM assessments a

        JOIN submissions sub
            ON a.submission_id = sub.id

        JOIN students s
            ON sub.student_id = s.id

        JOIN assignments ass
            ON sub.assignment_id = ass.id

        WHERE a.id = ?
        """,
        (assessment_id,),
    ).fetchone()

    if not assessment:
        conn.close()
        return None

    scores = conn.execute(
        """
        SELECT
            criterion_name,
            max_score,
            score,
            compliance,
            evidence,
            rationale,
            improvement
        FROM criterion_scores
        WHERE assessment_id = ?
        """,
        (assessment_id,),
    ).fetchall()

    feedback = conn.execute(
        """
        SELECT *
        FROM feedback
        WHERE assessment_id = ?
        """,
        (assessment_id,),
    ).fetchone()

    conn.close()

    result = dict(assessment)

    result["scores"] = [
        dict(score)
        for score in scores
    ]

    result["feedback"] = (
        dict(feedback)
        if feedback
        else None
    )

    return result

def get_all_assessments_for_vector_db():
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            a.id AS assessment_id,
            ass.prompt AS assignment_prompt,
            ass.assignment_type AS assignment_type,
            ass.rubric_text AS rubric,
            sub.content AS submission,
            f.summary AS feedback
        FROM assessments a

        JOIN submissions sub
            ON a.submission_id = sub.id

        JOIN assignments ass
            ON sub.assignment_id = ass.id

        LEFT JOIN feedback f
            ON f.assessment_id = a.id

        ORDER BY a.id
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]