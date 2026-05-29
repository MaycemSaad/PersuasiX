"""Social media monitoring routes."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from src.pipeline.social_monitor import SocialMonitor

router = APIRouter(prefix="/api/v1/social", tags=["Social Monitoring"])

_monitor: SocialMonitor | None = None


def _get_monitor() -> SocialMonitor:
    global _monitor
    if _monitor is None:
        _monitor = SocialMonitor()
    return _monitor


class SocialSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    platform: str = Field("reddit", pattern="^(twitter|reddit)$")
    limit: int = Field(20, ge=1, le=100)
    language: str = "en"
    analyze: bool = True


class SubredditMonitorRequest(BaseModel):
    subreddit: str = Field(..., min_length=2)
    sort: str = Field("hot", pattern="^(hot|new|top|rising)$")
    limit: int = Field(25, ge=1, le=100)
    analyze: bool = True


class MultiPlatformSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    platforms: list[str] = Field(default_factory=lambda: ["twitter", "reddit"])
    limit: int = Field(20, ge=1, le=100)


@router.get("/status")
def social_status():
    """Return which social platform connectors have credentials configured."""
    return _get_monitor().get_platform_status()


@router.post("/search")
def search_social(req: SocialSearchRequest):
    """Search Twitter/X or Reddit and optionally analyze returned posts."""
    monitor = _get_monitor()
    if req.platform == "twitter":
        results = monitor.search_twitter(req.query, req.limit, req.language, req.analyze)
    else:
        results = monitor.search_reddit(req.query, req.limit, req.analyze)
    return {"platform": req.platform, "count": len(results), "items": [_to_dict(r) for r in results]}


@router.post("/reddit/subreddit")
def monitor_subreddit(req: SubredditMonitorRequest):
    """Fetch and optionally analyze posts from a subreddit."""
    results = _get_monitor().monitor_subreddit(req.subreddit, req.sort, req.limit, req.analyze)
    return {"platform": "reddit", "subreddit": req.subreddit, "count": len(results), "items": [_to_dict(r) for r in results]}


@router.post("/multi-search")
def multi_platform_search(req: MultiPlatformSearchRequest):
    """Search configured social platforms for a topic."""
    results = _get_monitor().multi_platform_search(req.query, req.platforms, req.limit)
    return {
        platform: {"count": len(items), "items": [_to_dict(item) for item in items]}
        for platform, items in results.items()
    }


@router.get("/youtube/{video_id}/comments")
def analyze_youtube_comments(
    video_id: str,
    max_results: int = Query(50, ge=1, le=100),
    analyze: bool = True,
):
    """Fetch and optionally analyze YouTube comments for a video."""
    results = _get_monitor().analyze_youtube_comments(video_id, max_results, analyze)
    return {"platform": "youtube", "video_id": video_id, "count": len(results), "items": [_to_dict(r) for r in results]}


def _to_dict(item):
    return item.to_dict() if hasattr(item, "to_dict") else item
