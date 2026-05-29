"""
Collaborative annotation interface for PersuasiX.

Allows multiple annotators to:
  - Label text spans with persuasion techniques
  - Rate analysis quality (agree/disagree with model predictions)
  - Add custom explanations and corrections
  - Track inter-annotator agreement
  - Export annotations for model fine-tuning

Storage: SQLite-based annotation database (separate from main DB).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


ANNOTATIONS_DB = Path(__file__).resolve().parent.parent / "data" / "annotations.db"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Annotation:
    """A single annotation by an annotator."""
    id: int = 0
    text_id: str = ""
    annotator: str = ""
    text: str = ""
    # Span-level annotations
    spans: list[dict] = field(default_factory=list)
    # Document-level labels
    is_persuasive: bool | None = None
    techniques: list[str] = field(default_factory=list)
    severity: float | None = None
    # Feedback on model predictions
    model_agreement: str = ""  # agree, partial, disagree
    correction_notes: str = ""
    # Meta
    created_at: str = ""
    time_spent_seconds: float = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text_id": self.text_id,
            "annotator": self.annotator,
            "text": self.text[:200],
            "spans": self.spans,
            "is_persuasive": self.is_persuasive,
            "techniques": self.techniques,
            "severity": self.severity,
            "model_agreement": self.model_agreement,
            "correction_notes": self.correction_notes,
            "created_at": self.created_at,
            "time_spent_seconds": round(self.time_spent_seconds, 1),
        }


@dataclass
class AnnotationTask:
    """A text to be annotated."""
    text_id: str
    text: str
    language: str = "en"
    source: str = ""
    model_prediction: dict = field(default_factory=dict)
    annotations: list[Annotation] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, disputed

    def to_dict(self) -> dict:
        return {
            "text_id": self.text_id,
            "text": self.text[:200],
            "language": self.language,
            "source": self.source,
            "model_prediction": self.model_prediction,
            "num_annotations": len(self.annotations),
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class AnnotationDB:
    """SQLite-based annotation storage."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path or ANNOTATIONS_DB)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                text_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                language TEXT DEFAULT 'en',
                source TEXT DEFAULT '',
                model_prediction TEXT DEFAULT '{}',
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id TEXT NOT NULL,
                annotator TEXT NOT NULL,
                spans TEXT DEFAULT '[]',
                is_persuasive INTEGER,
                techniques TEXT DEFAULT '[]',
                severity REAL,
                model_agreement TEXT DEFAULT '',
                correction_notes TEXT DEFAULT '',
                time_spent_seconds REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (text_id) REFERENCES tasks(text_id)
            );

            CREATE TABLE IF NOT EXISTS annotators (
                name TEXT PRIMARY KEY,
                email TEXT DEFAULT '',
                role TEXT DEFAULT 'annotator',
                total_annotations INTEGER DEFAULT 0,
                avg_agreement REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_annotations_text ON annotations(text_id);
            CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations(annotator);
        """)
        conn.commit()
        conn.close()

    # ---- Tasks ----

    def add_task(self, text_id: str, text: str, language: str = "en",
                 source: str = "", model_prediction: dict | None = None) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR IGNORE INTO tasks (text_id, text, language, source, model_prediction) VALUES (?,?,?,?,?)",
            (text_id, text, language, source, json.dumps(model_prediction or {})),
        )
        conn.commit()
        conn.close()

    def add_tasks_bulk(self, tasks: list[dict]) -> int:
        """Add multiple tasks at once."""
        conn = sqlite3.connect(self.db_path)
        count = 0
        for t in tasks:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO tasks (text_id, text, language, source, model_prediction) VALUES (?,?,?,?,?)",
                    (t["text_id"], t["text"], t.get("language", "en"),
                     t.get("source", ""), json.dumps(t.get("model_prediction", {}))),
                )
                count += 1
            except Exception:
                continue
        conn.commit()
        conn.close()
        return count

    def get_tasks(self, status: str | None = None, limit: int = 50) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_next_task(self, annotator: str) -> dict | None:
        """Get the next unannotated task for a specific annotator."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT t.* FROM tasks t
            WHERE t.status IN ('pending', 'in_progress')
            AND t.text_id NOT IN (
                SELECT text_id FROM annotations WHERE annotator=?
            )
            ORDER BY t.created_at ASC LIMIT 1
        """, (annotator,)).fetchone()
        conn.close()
        return dict(row) if row else None

    # ---- Annotations ----

    def save_annotation(self, annotation: Annotation) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """INSERT INTO annotations
               (text_id, annotator, spans, is_persuasive, techniques, severity,
                model_agreement, correction_notes, time_spent_seconds)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (annotation.text_id, annotation.annotator, json.dumps(annotation.spans),
             1 if annotation.is_persuasive else 0 if annotation.is_persuasive is not None else None,
             json.dumps(annotation.techniques), annotation.severity,
             annotation.model_agreement, annotation.correction_notes,
             annotation.time_spent_seconds),
        )
        annotation_id = cursor.lastrowid

        # Update task status
        count = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE text_id=?",
            (annotation.text_id,),
        ).fetchone()[0]
        new_status = "completed" if count >= 2 else "in_progress"
        conn.execute(
            "UPDATE tasks SET status=?, updated_at=datetime('now') WHERE text_id=?",
            (new_status, annotation.text_id),
        )

        # Update annotator stats
        conn.execute(
            """INSERT INTO annotators (name, total_annotations) VALUES (?, 1)
               ON CONFLICT(name) DO UPDATE SET total_annotations = total_annotations + 1""",
            (annotation.annotator,),
        )

        conn.commit()
        conn.close()
        return annotation_id

    def get_annotations(self, text_id: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM annotations WHERE text_id=? ORDER BY created_at",
            (text_id,),
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            d["spans"] = json.loads(d.get("spans", "[]"))
            d["techniques"] = json.loads(d.get("techniques", "[]"))
            result.append(d)
        return result

    # ---- Inter-annotator agreement ----

    def compute_agreement(self, text_id: str) -> dict:
        """Compute inter-annotator agreement for a text."""
        annotations = self.get_annotations(text_id)
        if len(annotations) < 2:
            return {"agreement": None, "num_annotators": len(annotations)}

        # Binary agreement on is_persuasive
        verdicts = [a.get("is_persuasive") for a in annotations if a.get("is_persuasive") is not None]
        if verdicts:
            agreement_pct = sum(1 for v in verdicts if v == verdicts[0]) / len(verdicts)
        else:
            agreement_pct = 0

        # Technique overlap (Jaccard similarity)
        technique_sets = [set(a.get("techniques", [])) for a in annotations]
        if len(technique_sets) >= 2:
            intersection = technique_sets[0].intersection(technique_sets[1])
            union = technique_sets[0].union(technique_sets[1])
            jaccard = len(intersection) / max(len(union), 1)
        else:
            jaccard = 0

        return {
            "num_annotators": len(annotations),
            "verdict_agreement": round(agreement_pct, 3),
            "technique_jaccard": round(jaccard, 3),
            "model_agreements": [a.get("model_agreement", "") for a in annotations],
        }

    # ---- Statistics ----

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        total_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        completed = conn.execute("SELECT COUNT(*) FROM tasks WHERE status='completed'").fetchone()[0]
        total_annotations = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        annotators = conn.execute("SELECT COUNT(*) FROM annotators").fetchone()[0]

        agreement_counts = {"agree": 0, "partial": 0, "disagree": 0}
        rows = conn.execute("SELECT model_agreement, COUNT(*) FROM annotations GROUP BY model_agreement").fetchall()
        for row in rows:
            if row[0] in agreement_counts:
                agreement_counts[row[0]] = row[1]

        conn.close()
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "pending_tasks": total_tasks - completed,
            "total_annotations": total_annotations,
            "num_annotators": annotators,
            "model_agreement_distribution": agreement_counts,
            "avg_annotations_per_task": round(total_annotations / max(total_tasks, 1), 1),
        }

    # ---- Export ----

    def export_for_training(self, output_path: str, min_annotations: int = 2) -> int:
        """Export annotations as JSONL for model fine-tuning."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        tasks = conn.execute(
            "SELECT * FROM tasks WHERE status='completed'"
        ).fetchall()

        exported = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for task in tasks:
                annotations = self.get_annotations(task["text_id"])
                if len(annotations) < min_annotations:
                    continue

                # Majority vote on techniques
                all_techniques: dict[str, int] = {}
                verdicts = []
                severities = []

                for a in annotations:
                    verdicts.append(a.get("is_persuasive"))
                    if a.get("severity") is not None:
                        severities.append(a["severity"])
                    for t in a.get("techniques", []):
                        all_techniques[t] = all_techniques.get(t, 0) + 1

                majority_threshold = len(annotations) / 2
                final_techniques = [t for t, c in all_techniques.items() if c > majority_threshold]
                final_persuasive = sum(1 for v in verdicts if v) > majority_threshold
                final_severity = sum(severities) / max(len(severities), 1) if severities else 0

                # Merge spans from all annotators
                all_spans = []
                for a in annotations:
                    all_spans.extend(a.get("spans", []))

                record = {
                    "text": task["text"],
                    "language": task["language"],
                    "techniques": final_techniques,
                    "is_persuasive": final_persuasive,
                    "severity": round(final_severity, 1),
                    "spans": all_spans,
                    "num_annotators": len(annotations),
                    "source": "annotation",
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        conn.close()
        logger.info(f"Exported {exported} annotated examples to {output_path}")
        return exported


# ---------------------------------------------------------------------------
# Annotation API routes (to add to FastAPI)
# ---------------------------------------------------------------------------

def create_annotation_routes():
    """Create FastAPI routes for the annotation interface."""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field

    router = APIRouter(prefix="/api/v1/annotate", tags=["Annotation"])
    db: AnnotationDB | None = None

    def get_annotation_db() -> AnnotationDB:
        nonlocal db
        if db is None:
            db = AnnotationDB()
        return db

    class TaskCreateRequest(BaseModel):
        text_id: str
        text: str
        language: str = "en"
        source: str = ""
        model_prediction: dict = {}

    class AnnotationRequest(BaseModel):
        text_id: str
        annotator: str
        spans: list[dict] = []
        is_persuasive: bool | None = None
        techniques: list[str] = []
        severity: float | None = None
        model_agreement: str = ""
        correction_notes: str = ""
        time_spent_seconds: float = 0

    @router.post("/tasks")
    def create_task(req: TaskCreateRequest):
        db = get_annotation_db()
        db.add_task(req.text_id, req.text, req.language, req.source, req.model_prediction)
        return {"status": "created", "text_id": req.text_id}

    @router.post("/tasks/bulk")
    def create_tasks_bulk(tasks: list[TaskCreateRequest]):
        db = get_annotation_db()
        count = db.add_tasks_bulk([t.dict() for t in tasks])
        return {"status": "created", "count": count}

    @router.get("/tasks")
    def list_tasks(status: str | None = None, limit: int = 50):
        db = get_annotation_db()
        return db.get_tasks(status, limit)

    @router.get("/tasks/next")
    def next_task(annotator: str):
        db = get_annotation_db()
        task = db.get_next_task(annotator)
        if not task:
            raise HTTPException(404, "No pending tasks")
        return task

    @router.post("/submit")
    def submit_annotation(req: AnnotationRequest):
        db = get_annotation_db()
        annotation = Annotation(
            text_id=req.text_id,
            annotator=req.annotator,
            spans=req.spans,
            is_persuasive=req.is_persuasive,
            techniques=req.techniques,
            severity=req.severity,
            model_agreement=req.model_agreement,
            correction_notes=req.correction_notes,
            time_spent_seconds=req.time_spent_seconds,
        )
        ann_id = db.save_annotation(annotation)
        return {"status": "saved", "annotation_id": ann_id}

    @router.get("/agreement/{text_id}")
    def get_agreement(text_id: str):
        db = get_annotation_db()
        return db.compute_agreement(text_id)

    @router.get("/stats")
    def annotation_stats():
        db = get_annotation_db()
        return db.get_stats()

    @router.post("/export")
    def export_annotations(output_path: str = "data/annotated_export.jsonl", min_annotations: int = 2):
        db = get_annotation_db()
        count = db.export_for_training(output_path, min_annotations)
        return {"status": "exported", "count": count, "path": output_path}

    return router
