"""RSS/News feed monitoring routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.database import FeedAlert, MonitoredFeed, get_db
from api.schemas import AlertResponse, FeedCreateRequest, FeedResponse

router = APIRouter(prefix="/api/v1/monitor", tags=["Monitor"])


@router.post("/feeds", response_model=FeedResponse)
def add_feed(req: FeedCreateRequest, db: Session = Depends(get_db)):
    """Add a new RSS feed to monitor."""
    existing = db.query(MonitoredFeed).filter(MonitoredFeed.url == req.url).first()
    if existing:
        raise HTTPException(status_code=409, detail="Feed already exists")

    feed = MonitoredFeed(
        url=req.url,
        name=req.name or req.url,
        language=req.language,
        check_interval_min=req.check_interval_min,
    )
    db.add(feed)
    db.commit()
    db.refresh(feed)
    return feed.to_dict()


@router.get("/feeds", response_model=list[FeedResponse])
def list_feeds(db: Session = Depends(get_db)):
    """List all monitored feeds."""
    feeds = db.query(MonitoredFeed).order_by(MonitoredFeed.created_at.desc()).all()
    return [f.to_dict() for f in feeds]


@router.delete("/feeds/{feed_id}")
def delete_feed(feed_id: int, db: Session = Depends(get_db)):
    """Remove a monitored feed."""
    feed = db.query(MonitoredFeed).filter(MonitoredFeed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    db.delete(feed)
    db.commit()
    return {"detail": "Feed removed"}


@router.patch("/feeds/{feed_id}/toggle")
def toggle_feed(feed_id: int, db: Session = Depends(get_db)):
    """Toggle a feed on/off."""
    feed = db.query(MonitoredFeed).filter(MonitoredFeed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    feed.active = not feed.active
    db.commit()
    return {"active": feed.active}


@router.get("/alerts", response_model=list[AlertResponse])
def list_alerts(
    unread_only: bool = False,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List manipulation alerts from monitored feeds."""
    query = db.query(FeedAlert).order_by(FeedAlert.created_at.desc())
    if unread_only:
        query = query.filter(FeedAlert.read == False)
    alerts = query.limit(limit).all()
    return [a.to_dict() for a in alerts]


@router.patch("/alerts/{alert_id}/read")
def mark_alert_read(alert_id: int, db: Session = Depends(get_db)):
    """Mark an alert as read."""
    alert = db.query(FeedAlert).filter(FeedAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.read = True
    db.commit()
    return {"detail": "Alert marked as read"}


@router.post("/check-now")
def check_feeds_now():
    """Manually trigger a feed check."""
    from monitor.rss_monitor import check_all_feeds
    results = check_all_feeds()
    return {"detail": f"Checked {results['feeds_checked']} feeds, found {results['new_alerts']} new alerts"}
