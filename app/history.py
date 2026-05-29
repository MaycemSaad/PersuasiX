"""Analysis history storage and statistics for PersuasiX Dashboard."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "analysis_history.json"


def _ensure_file():
    """Create history file if it doesn't exist."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_history() -> list[dict]:
    """Load analysis history from disk."""
    _ensure_file()
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_analysis(result: dict, source_type: str = "text", source_label: str = "") -> None:
    """Save an analysis result to history."""
    _ensure_file()
    history = load_history()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "source_type": source_type,
        "source_label": source_label or result.get("text", "")[:80],
        "is_persuasive": result.get("is_persuasive", False),
        "techniques": result.get("techniques", []),
        "technique_count": len(result.get("techniques", [])),
        "severity_score": result.get("severity_score", 0),
        "manipulation_score": result.get("manipulation_score", 0),
        "language": result.get("language", "en"),
    }

    history.append(entry)

    # Keep only last 200 entries
    if len(history) > 200:
        history = history[-200:]

    try:
        HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning(f"Could not save history: {e}")


def get_stats() -> dict[str, Any]:
    """Compute aggregate statistics from history."""
    history = load_history()

    if not history:
        return {
            "total_analyses": 0,
            "manipulative_count": 0,
            "neutral_count": 0,
            "avg_severity": 0,
            "avg_manipulation": 0,
            "technique_frequency": {},
            "source_breakdown": {},
            "language_breakdown": {},
            "recent": [],
        }

    manipulative = [h for h in history if h.get("is_persuasive")]
    neutral = [h for h in history if not h.get("is_persuasive")]

    # Technique frequency
    tech_freq: dict[str, int] = {}
    for h in history:
        for t in h.get("techniques", []):
            tech_freq[t] = tech_freq.get(t, 0) + 1

    # Source breakdown
    source_freq: dict[str, int] = {}
    for h in history:
        st = h.get("source_type", "text")
        source_freq[st] = source_freq.get(st, 0) + 1

    # Language breakdown
    lang_freq: dict[str, int] = {}
    for h in history:
        lang = h.get("language", "en")
        lang_freq[lang] = lang_freq.get(lang, 0) + 1

    severities = [h.get("severity_score", 0) for h in manipulative]
    manip_scores = [h.get("manipulation_score", 0) for h in manipulative]

    return {
        "total_analyses": len(history),
        "manipulative_count": len(manipulative),
        "neutral_count": len(neutral),
        "avg_severity": sum(severities) / len(severities) if severities else 0,
        "avg_manipulation": sum(manip_scores) / len(manip_scores) if manip_scores else 0,
        "technique_frequency": dict(sorted(tech_freq.items(), key=lambda x: -x[1])),
        "source_breakdown": source_freq,
        "language_breakdown": lang_freq,
        "recent": history[-10:][::-1],
    }


def clear_history() -> None:
    """Clear all analysis history."""
    _ensure_file()
    HISTORY_FILE.write_text("[]", encoding="utf-8")
    logger.info("Analysis history cleared")
