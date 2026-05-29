"""Advanced intelligence utilities for PersuasiX.

This module turns stored analysis history into operational signals:
  - active-learning candidates for human review
  - dataset/model drift reports
  - narrative clusters for coordinated-message discovery
  - compact threat intelligence summaries
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _safe_json(value: str, fallback):
    try:
        return json.loads(value or "")
    except Exception:
        return fallback


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z_-]{2,}", text.lower())


def _entropy(probs: dict[str, float]) -> float:
    values = [max(min(float(v), 1.0), 0.0) for v in probs.values()]
    if not values:
        return 0.0
    total = 0.0
    for p in values:
        if p in (0.0, 1.0):
            continue
        total += -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return total / max(len(values), 1)


def _jensen_shannon(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    p_sum = sum(p.get(k, 0.0) for k in keys) or 1.0
    q_sum = sum(q.get(k, 0.0) for k in keys) or 1.0
    p_norm = {k: p.get(k, 0.0) / p_sum for k in keys}
    q_norm = {k: q.get(k, 0.0) / q_sum for k in keys}
    mid = {k: (p_norm[k] + q_norm[k]) / 2 for k in keys}
    return round((_kl(p_norm, mid) + _kl(q_norm, mid)) / 2, 4)


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    total = 0.0
    for key, val in p.items():
        if val <= 0:
            continue
        total += val * math.log2(val / max(q.get(key, 0.0), 1e-12))
    return total


def _distribution(rows: list[dict], field: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if isinstance(value, list):
            counts.update(value)
        elif value:
            counts[str(value)] += 1
    total = sum(counts.values()) or 1
    return {k: round(v / total, 4) for k, v in counts.items()}


def _avg(values: list[float]) -> float:
    return round(sum(values) / max(len(values), 1), 4)


@dataclass
class ActiveLearningCandidate:
    """A stored analysis that should be reviewed by a human annotator."""

    analysis_id: int
    text_preview: str
    uncertainty_score: float
    reasons: list[str] = field(default_factory=list)
    techniques: list[str] = field(default_factory=list)
    manipulation_score: float = 0.0
    severity_score: float = 0.0
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "text_preview": self.text_preview,
            "uncertainty_score": round(self.uncertainty_score, 4),
            "reasons": self.reasons,
            "techniques": self.techniques,
            "manipulation_score": round(self.manipulation_score, 4),
            "severity_score": round(self.severity_score, 2),
            "created_at": self.created_at,
        }


class PersuasixIntelligence:
    """Operational analytics over PersuasiX analysis records."""

    def __init__(self, analyses: list[Any]) -> None:
        self.rows = [self._normalize(row) for row in analyses]

    @classmethod
    def from_database(cls, db, limit: int = 1000) -> "PersuasixIntelligence":
        from api.database import Analysis

        rows = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
        return cls(rows)

    def active_learning_queue(self, limit: int = 25) -> dict[str, Any]:
        """Rank records that are likely to improve model quality if reviewed."""
        candidates = []
        for row in self.rows:
            probs = row["technique_probabilities"]
            entropy = _entropy(probs)
            score = float(row["manipulation_score"])
            severity = float(row["severity_score"])
            reasons = []

            near_boundary = 1.0 - min(abs(score - 0.5) * 2, 1.0)
            if near_boundary > 0.65:
                reasons.append("manipulation score near decision boundary")

            if entropy > 0.65:
                reasons.append("high multi-label probability entropy")

            if row["is_persuasive"] and not row["techniques"]:
                reasons.append("persuasive verdict without technique labels")

            if row["techniques"] and score < 0.35:
                reasons.append("techniques detected but global score is low")

            if severity >= 4 and score < 0.55:
                reasons.append("high severity but moderate manipulation score")

            uncertainty = round((0.45 * entropy) + (0.35 * near_boundary) + (0.20 * min(len(reasons) / 3, 1)), 4)
            if uncertainty <= 0 and not reasons:
                continue

            candidates.append(ActiveLearningCandidate(
                analysis_id=row["id"],
                text_preview=row["text"][:220],
                uncertainty_score=uncertainty,
                reasons=reasons or ["low-confidence review candidate"],
                techniques=row["techniques"],
                manipulation_score=score,
                severity_score=severity,
                created_at=row["created_at"],
            ))

        candidates.sort(key=lambda c: c.uncertainty_score, reverse=True)
        return {
            "total_records_scanned": len(self.rows),
            "candidate_count": min(len(candidates), limit),
            "items": [c.to_dict() for c in candidates[:limit]],
        }

    def drift_report(self, baseline_size: int = 200, recent_size: int = 100) -> dict[str, Any]:
        """Compare recent analysis distributions against older baseline records."""
        if len(self.rows) < 2:
            return {"status": "insufficient_data", "total_records": len(self.rows)}

        recent = self.rows[:recent_size]
        baseline = self.rows[recent_size:recent_size + baseline_size] or self.rows[recent_size:]
        if not baseline:
            midpoint = max(len(self.rows) // 2, 1)
            recent = self.rows[:midpoint]
            baseline = self.rows[midpoint:]

        technique_drift = _jensen_shannon(
            _distribution(recent, "techniques"),
            _distribution(baseline, "techniques"),
        )
        language_drift = _jensen_shannon(
            _distribution(recent, "language"),
            _distribution(baseline, "language"),
        )
        source_drift = _jensen_shannon(
            _distribution(recent, "source_type"),
            _distribution(baseline, "source_type"),
        )

        recent_scores = [float(r["manipulation_score"]) for r in recent]
        baseline_scores = [float(r["manipulation_score"]) for r in baseline]
        score_shift = round(_avg(recent_scores) - _avg(baseline_scores), 4)

        max_drift = max(technique_drift, language_drift, source_drift, abs(score_shift))
        status = "stable"
        if max_drift >= 0.2:
            status = "attention"
        if max_drift >= 0.35:
            status = "drift_detected"

        return {
            "status": status,
            "total_records": len(self.rows),
            "recent_window": len(recent),
            "baseline_window": len(baseline),
            "technique_js_divergence": technique_drift,
            "language_js_divergence": language_drift,
            "source_js_divergence": source_drift,
            "score_shift": score_shift,
            "recent_avg_manipulation": _avg(recent_scores),
            "baseline_avg_manipulation": _avg(baseline_scores),
            "recent_top_techniques": self._top_terms(recent, "techniques", 8),
            "baseline_top_techniques": self._top_terms(baseline, "techniques", 8),
        }

    def narrative_clusters(self, limit: int = 8, min_cluster_size: int = 2) -> dict[str, Any]:
        """Find repeated narratives using lightweight lexical clustering."""
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in self.rows:
            tokens = [t for t in _tokenize(row["text"]) if t not in STOPWORDS]
            if not tokens:
                continue
            key_terms = [term for term, _ in Counter(tokens).most_common(4)]
            key = " ".join(sorted(key_terms[:3]))
            if key:
                buckets[key].append(row)

        clusters = []
        for key, items in buckets.items():
            if len(items) < min_cluster_size:
                continue
            techniques = Counter(t for item in items for t in item["techniques"])
            scores = [float(item["manipulation_score"]) for item in items]
            clusters.append({
                "signature": key,
                "size": len(items),
                "avg_manipulation_score": _avg(scores),
                "top_techniques": dict(techniques.most_common(5)),
                "examples": [
                    {
                        "analysis_id": item["id"],
                        "text_preview": item["text"][:180],
                        "source_type": item["source_type"],
                        "created_at": item["created_at"],
                    }
                    for item in items[:5]
                ],
            })

        clusters.sort(key=lambda c: (c["size"], c["avg_manipulation_score"]), reverse=True)
        return {
            "cluster_count": min(len(clusters), limit),
            "items": clusters[:limit],
        }

    def threat_report(self) -> dict[str, Any]:
        """Build a compact operational summary for analysts."""
        total = len(self.rows)
        persuasive = [r for r in self.rows if r["is_persuasive"]]
        high_risk = [r for r in persuasive if float(r["manipulation_score"]) >= 0.7 or float(r["severity_score"]) >= 4]
        all_techniques = Counter(t for r in persuasive for t in r["techniques"])
        sources = Counter(r["source_type"] for r in self.rows)

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_records": total,
            "persuasive_rate": round(len(persuasive) / max(total, 1), 4),
            "high_risk_count": len(high_risk),
            "avg_manipulation_score": _avg([float(r["manipulation_score"]) for r in self.rows]),
            "top_techniques": dict(all_techniques.most_common(10)),
            "source_breakdown": dict(sources),
            "active_learning": self.active_learning_queue(limit=5),
            "drift": self.drift_report(),
            "narrative_clusters": self.narrative_clusters(limit=5),
        }

    def _normalize(self, row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return {
                "id": row.get("id", 0),
                "text": row.get("text") or row.get("source_label", ""),
                "language": row.get("language", "en"),
                "is_persuasive": bool(row.get("is_persuasive", False)),
                "techniques": row.get("techniques", []),
                "technique_probabilities": row.get("technique_probabilities", {}),
                "severity_score": float(row.get("severity_score", 0.0) or 0.0),
                "manipulation_score": float(row.get("manipulation_score", 0.0) or 0.0),
                "source_type": row.get("source_type", "text"),
                "source_label": row.get("source_label", ""),
                "created_at": row.get("created_at") or row.get("timestamp"),
            }
        return {
            "id": row.id,
            "text": row.text or "",
            "language": row.language or "en",
            "is_persuasive": bool(row.is_persuasive),
            "techniques": _safe_json(row.techniques_json, []),
            "technique_probabilities": _safe_json(row.technique_probs_json, {}),
            "severity_score": float(row.severity_score or 0.0),
            "manipulation_score": float(row.manipulation_score or 0.0),
            "source_type": row.source_type or "text",
            "source_label": row.source_label or "",
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    def _top_terms(self, rows: list[dict], field: str, limit: int) -> dict[str, int]:
        counts = Counter()
        for row in rows:
            value = row.get(field)
            if isinstance(value, list):
                counts.update(value)
            elif value:
                counts[str(value)] += 1
        return dict(counts.most_common(limit))


STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "you", "your",
    "have", "has", "was", "were", "will", "would", "about", "only", "they",
    "their", "there", "because", "into", "over", "under", "not", "but", "our",
    "all", "can", "cannot", "should", "must", "more", "less", "than", "then",
}
