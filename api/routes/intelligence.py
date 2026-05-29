"""Advanced intelligence endpoints for analysts and model operators."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.database import get_db
from src.pipeline.intelligence import PersuasixIntelligence

router = APIRouter(prefix="/api/v1/intelligence", tags=["Intelligence"])


def _intel(db: Session, limit: int) -> PersuasixIntelligence:
    return PersuasixIntelligence.from_database(db, limit=limit)


@router.get("/active-learning")
def active_learning_queue(
    limit: int = Query(25, ge=1, le=100),
    scan_limit: int = Query(1000, ge=10, le=5000),
    db: Session = Depends(get_db),
):
    """Return analyses that are most valuable for human review."""
    return _intel(db, scan_limit).active_learning_queue(limit=limit)


@router.get("/drift")
def drift_report(
    baseline_size: int = Query(200, ge=10, le=2000),
    recent_size: int = Query(100, ge=5, le=1000),
    scan_limit: int = Query(2000, ge=20, le=5000),
    db: Session = Depends(get_db),
):
    """Compare recent analysis distributions against older baseline records."""
    return _intel(db, scan_limit).drift_report(
        baseline_size=baseline_size,
        recent_size=recent_size,
    )


@router.get("/narrative-clusters")
def narrative_clusters(
    limit: int = Query(8, ge=1, le=50),
    min_cluster_size: int = Query(2, ge=2, le=20),
    scan_limit: int = Query(1000, ge=20, le=5000),
    db: Session = Depends(get_db),
):
    """Group repeated narratives and repeated message signatures."""
    return _intel(db, scan_limit).narrative_clusters(
        limit=limit,
        min_cluster_size=min_cluster_size,
    )


@router.get("/threat-report")
def threat_report(
    scan_limit: int = Query(1000, ge=20, le=5000),
    db: Session = Depends(get_db),
):
    """Return a compact analyst report combining risk, drift, and clusters."""
    return _intel(db, scan_limit).threat_report()
