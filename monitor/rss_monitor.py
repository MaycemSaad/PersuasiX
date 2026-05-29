"""Real-time RSS feed monitoring for persuasion/manipulation detection."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger


def check_all_feeds() -> dict:
    """Check all active feeds for new manipulative articles."""
    import feedparser
    from app.scraper import scrape_url
    from api.database import (
        Analysis, FeedAlert, MonitoredFeed, SessionLocal, save_analysis_to_db,
    )

    db = SessionLocal()
    feeds_checked = 0
    new_alerts = 0

    try:
        feeds = db.query(MonitoredFeed).filter(MonitoredFeed.active == True).all()
        if not feeds:
            logger.info("No active feeds to check")
            return {"feeds_checked": 0, "new_alerts": 0}

        # Load pipeline
        from src.pipeline.persuasix_pipeline import PersuasixPipeline
        pipeline = PersuasixPipeline.from_default_models(device="cpu")

        for feed in feeds:
            try:
                logger.info(f"Checking feed: {feed.name} ({feed.url})")
                parsed = feedparser.parse(feed.url)
                feeds_checked += 1

                for entry in parsed.entries[:10]:  # Last 10 entries
                    url = entry.get("link", "")
                    title = entry.get("title", "")

                    if not url:
                        continue

                    # Skip if already analyzed
                    existing = db.query(FeedAlert).filter(FeedAlert.article_url == url).first()
                    if existing:
                        continue

                    # Scrape and analyze
                    article = scrape_url(url)
                    if not article.success or not article.text.strip():
                        continue

                    text = article.text
                    words = text.split()
                    if len(words) > 2000:
                        text = " ".join(words[:2000])

                    result = pipeline.analyze(text, language=feed.language)

                    # Only alert if manipulation score > 40%
                    if result.manipulation_score > 0.4:
                        result_dict = result.to_dict()
                        record = save_analysis_to_db(
                            result_dict,
                            source_type="monitor",
                            source_label=title or article.source,
                        )

                        alert = FeedAlert(
                            feed_id=feed.id,
                            article_url=url,
                            article_title=title or article.title,
                            manipulation_score=result.manipulation_score,
                            severity_score=result.severity_score,
                            techniques_json=json.dumps(result.techniques),
                            analysis_id=record.id,
                        )
                        db.add(alert)
                        new_alerts += 1
                        logger.warning(
                            f"ALERT: {title[:60]} — Score: {int(result.manipulation_score*100)}% "
                            f"({len(result.techniques)} techniques)"
                        )

                feed.last_checked = datetime.utcnow()
                db.commit()

            except Exception as e:
                logger.error(f"Error checking feed {feed.name}: {e}")
                continue

    finally:
        db.close()

    logger.info(f"Feed check complete: {feeds_checked} feeds, {new_alerts} new alerts")
    return {"feeds_checked": feeds_checked, "new_alerts": new_alerts}


def start_monitor_scheduler():
    """Start the background scheduler for feed monitoring."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler()
        scheduler.add_job(
            check_all_feeds,
            "interval",
            minutes=15,
            id="feed_monitor",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Feed monitor scheduler started (every 15 min)")
        return scheduler
    except ImportError:
        logger.warning("apscheduler not installed — feed monitoring disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to start monitor scheduler: {e}")
        return None
