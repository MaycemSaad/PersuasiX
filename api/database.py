"""SQLite database layer with SQLAlchemy ORM for PersuasiX."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "persuasix.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    """Stores every analysis result."""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    language = Column(String(5), default="en")
    is_persuasive = Column(Boolean, default=False)
    techniques_json = Column(Text, default="[]")
    technique_probs_json = Column(Text, default="{}")
    highlighted_json = Column(Text, default="[]")
    explanation = Column(Text, default="")
    neutral_rewrite = Column(Text, default="")
    severity_score = Column(Float, default=0.0)
    manipulation_score = Column(Float, default=0.0)
    cross_lingual_json = Column(Text, default="{}")
    source_type = Column(String(20), default="text")   # text, url, file, batch, monitor
    source_label = Column(String(500), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    @property
    def techniques(self) -> list[str]:
        return json.loads(self.techniques_json) if self.techniques_json else []

    @property
    def highlighted_phrases(self) -> list[dict]:
        return json.loads(self.highlighted_json) if self.highlighted_json else []

    @property
    def cross_lingual(self) -> dict:
        return json.loads(self.cross_lingual_json) if self.cross_lingual_json else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "is_persuasive": self.is_persuasive,
            "techniques": self.techniques,
            "technique_probabilities": json.loads(self.technique_probs_json) if self.technique_probs_json else {},
            "highlighted_phrases": self.highlighted_phrases,
            "explanation": self.explanation,
            "neutral_rewrite": self.neutral_rewrite,
            "severity_score": self.severity_score,
            "manipulation_score": self.manipulation_score,
            "cross_lingual_explanations": self.cross_lingual,
            "source_type": self.source_type,
            "source_label": self.source_label,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MonitoredFeed(Base):
    """RSS/Atom feeds being monitored for manipulation."""
    __tablename__ = "monitored_feeds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(1000), nullable=False, unique=True)
    name = Column(String(200), default="")
    language = Column(String(5), default="en")
    check_interval_min = Column(Integer, default=30)
    last_checked = Column(DateTime, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "name": self.name,
            "language": self.language,
            "check_interval_min": self.check_interval_min,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "active": self.active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class FeedAlert(Base):
    """Alerts from monitored feeds — high-manipulation articles."""
    __tablename__ = "feed_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feed_id = Column(Integer, nullable=False)
    article_url = Column(String(1000), default="")
    article_title = Column(String(500), default="")
    manipulation_score = Column(Float, default=0.0)
    severity_score = Column(Float, default=0.0)
    techniques_json = Column(Text, default="[]")
    analysis_id = Column(Integer, nullable=True)
    read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    @property
    def techniques(self) -> list[str]:
        return json.loads(self.techniques_json) if self.techniques_json else []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "feed_id": self.feed_id,
            "article_url": self.article_url,
            "article_title": self.article_title,
            "manipulation_score": self.manipulation_score,
            "severity_score": self.severity_score,
            "techniques": self.techniques,
            "analysis_id": self.analysis_id,
            "read": self.read,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Report(Base):
    """Generated PDF reports."""
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, nullable=True)
    filename = Column(String(200), default="")
    filepath = Column(String(1000), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "filename": self.filename,
            "filepath": self.filepath,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables."""
    Base.metadata.create_all(engine)


def get_db() -> Session:
    """Dependency for FastAPI — yields a session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_analysis_to_db(result_dict: dict, source_type: str = "text", source_label: str = "") -> Analysis:
    """Save an analysis result to the database."""
    db = SessionLocal()
    try:
        record = Analysis(
            text=result_dict.get("text", ""),
            language=result_dict.get("language", "en"),
            is_persuasive=result_dict.get("is_persuasive", False),
            techniques_json=json.dumps(result_dict.get("techniques", [])),
            technique_probs_json=json.dumps(result_dict.get("technique_probabilities", {})),
            highlighted_json=json.dumps(result_dict.get("highlighted_phrases", [])),
            explanation=result_dict.get("explanation", ""),
            neutral_rewrite=result_dict.get("neutral_rewrite", ""),
            severity_score=result_dict.get("severity_score", 0.0),
            manipulation_score=result_dict.get("manipulation_score", 0.0),
            cross_lingual_json=json.dumps(result_dict.get("cross_lingual_explanations", {})),
            source_type=source_type,
            source_label=source_label or result_dict.get("text", "")[:80],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def get_db_stats() -> dict:
    """Compute aggregate statistics from the database."""
    db = SessionLocal()
    try:
        total = db.query(func.count(Analysis.id)).scalar() or 0
        manip = db.query(func.count(Analysis.id)).filter(Analysis.is_persuasive == True).scalar() or 0
        neutral = total - manip
        avg_sev = db.query(func.avg(Analysis.severity_score)).filter(Analysis.is_persuasive == True).scalar() or 0
        avg_manip = db.query(func.avg(Analysis.manipulation_score)).filter(Analysis.is_persuasive == True).scalar() or 0

        # Technique frequency
        all_analyses = db.query(Analysis.techniques_json).filter(Analysis.is_persuasive == True).all()
        tech_freq: dict[str, int] = {}
        for (tj,) in all_analyses:
            for t in json.loads(tj or "[]"):
                tech_freq[t] = tech_freq.get(t, 0) + 1

        # Source breakdown
        source_counts = db.query(Analysis.source_type, func.count(Analysis.id)).group_by(Analysis.source_type).all()

        # Recent
        recent = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(10).all()

        # Unread alerts
        unread_alerts = db.query(func.count(FeedAlert.id)).filter(FeedAlert.read == False).scalar() or 0

        return {
            "total_analyses": total,
            "manipulative_count": manip,
            "neutral_count": neutral,
            "avg_severity": round(avg_sev, 2),
            "avg_manipulation": round(avg_manip, 4),
            "technique_frequency": dict(sorted(tech_freq.items(), key=lambda x: -x[1])),
            "source_breakdown": {s: c for s, c in source_counts},
            "recent": [a.to_dict() for a in recent],
            "unread_alerts": unread_alerts,
        }
    finally:
        db.close()


# Initialize tables on import
init_db()
