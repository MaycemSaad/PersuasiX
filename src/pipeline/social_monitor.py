"""
Social media integration for real-time persuasion monitoring.

Supports:
  FREE (no API key needed):
  - Mastodon (Public timeline streaming, multi-instance)
  - RSS Feeds (Major news outlets worldwide)
  - YouTube Transcripts (via youtube-transcript-api, no key)
  - Hacker News (Firebase API, real-time)

  PAID (requires API keys):
  - Twitter/X API v2 (stream + search)
  - Reddit API (PRAW/OAuth)
  - YouTube Comments (via Data API v3)

Each connector:
  1. Fetches posts/comments matching keywords or from specific sources
  2. Runs PersuasiX analysis on each
  3. Stores results + triggers alerts for high-score content

Usage:
    from src.pipeline.social_monitor import SocialMonitor
    monitor = SocialMonitor()

    # FREE - No API keys needed
    monitor.search_mastodon("climate change", limit=30)
    monitor.stream_mastodon(keywords=["propaganda"], duration=60)
    monitor.fetch_rss_news(languages=["en", "fr", "ar"])
    monitor.analyze_youtube_video("dQw4w9WgXcQ")
    monitor.fetch_hackernews(limit=20)

    # Multi-platform search (free sources only)
    monitor.multi_platform_search("immigration", platforms=["mastodon", "rss", "hackernews"])
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import requests
import feedparser
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SocialPost:
    """A single social media post/comment."""
    platform: str           # mastodon, rss, youtube, hackernews, twitter, reddit
    post_id: str
    text: str
    author: str = ""
    url: str = ""
    timestamp: str = ""
    language: str = "en"
    engagement: dict = field(default_factory=dict)  # likes, boosts, score, etc.
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "post_id": self.post_id,
            "text": self.text[:500],
            "author": self.author,
            "url": self.url,
            "timestamp": self.timestamp,
            "language": self.language,
            "engagement": self.engagement,
            "metadata": self.metadata,
        }


@dataclass
class SocialAnalysisResult:
    """Analysis result for a social media post."""
    post: SocialPost
    is_persuasive: bool = False
    manipulation_score: float = 0.0
    severity_score: float = 0.0
    techniques: list[str] = field(default_factory=list)
    highlighted_phrases: list[dict] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "post": self.post.to_dict(),
            "is_persuasive": self.is_persuasive,
            "manipulation_score": round(self.manipulation_score, 4),
            "severity_score": round(self.severity_score, 2),
            "techniques": self.techniques,
            "highlighted_phrases": self.highlighted_phrases,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# MASTODON Connector (FREE - No auth required for public timelines)
# ---------------------------------------------------------------------------

class MastodonConnector:
    """
    Connect to Mastodon/Fediverse instances for real-time monitoring.

    Features:
      - No API key needed for public timelines
      - Multi-instance support (mastodon.social, mas.to, fosstodon.org, etc.)
      - Real-time streaming via Server-Sent Events (SSE)
      - Search by keywords/hashtags
      - Multilingual content (en, fr, de, es, ar, etc.)
      - 300 requests / 5 minutes per instance (generous)
    """

    DEFAULT_INSTANCES = [
        "https://mastodon.social",
        "https://mas.to",
        "https://fosstodon.org",
        "https://mastodon.online",
        "https://mstdn.social",
    ]

    def __init__(self, instances: list[str] | None = None) -> None:
        self.instances = instances or self.DEFAULT_INSTANCES
        self._streaming = False
        logger.info(f"Mastodon connector ready ({len(self.instances)} instances)")

    @property
    def is_available(self) -> bool:
        return True  # Always available, no auth needed

    def get_public_timeline(
        self,
        instance: str | None = None,
        limit: int = 40,
        local: bool = False,
        min_text_length: int = 30,
    ) -> list[SocialPost]:
        """Fetch public timeline from a Mastodon instance."""
        instance = instance or self.instances[0]
        posts = []

        try:
            resp = requests.get(
                f"{instance}/api/v1/timelines/public",
                params={"limit": min(limit, 40), "local": local},
                timeout=10,
            )

            if resp.status_code == 200:
                for toot in resp.json():
                    # Strip HTML tags
                    content = re.sub(r'<[^>]+>', '', toot.get("content", "")).strip()
                    if len(content) < min_text_length:
                        continue

                    lang = toot.get("language") or "en"
                    account = toot.get("account", {})

                    posts.append(SocialPost(
                        platform="mastodon",
                        post_id=toot.get("id", ""),
                        text=content,
                        author=account.get("acct", account.get("username", "")),
                        url=toot.get("url", ""),
                        timestamp=toot.get("created_at", ""),
                        language=lang,
                        engagement={
                            "reblogs": toot.get("reblogs_count", 0),
                            "favourites": toot.get("favourites_count", 0),
                            "replies": toot.get("replies_count", 0),
                        },
                        metadata={
                            "instance": urlparse(instance).netloc,
                            "visibility": toot.get("visibility", "public"),
                            "sensitive": toot.get("sensitive", False),
                            "tags": [t.get("name", "") for t in toot.get("tags", [])],
                        },
                    ))

                logger.debug(f"Mastodon {instance}: {len(posts)} posts fetched")
            else:
                logger.warning(f"Mastodon {instance}: HTTP {resp.status_code}")

        except Exception as e:
            logger.error(f"Mastodon {instance} failed: {e}")

        return posts

    def search(
        self,
        query: str,
        limit: int = 30,
        language: str | None = None,
    ) -> list[SocialPost]:
        """Search across multiple Mastodon instances."""
        all_posts = []

        for instance in self.instances:
            try:
                # Use hashtag search (no auth) or public timeline filtering
                resp = requests.get(
                    f"{instance}/api/v1/timelines/tag/{query.replace(' ', '').lower()}",
                    params={"limit": min(limit // len(self.instances) + 5, 40)},
                    timeout=10,
                )

                if resp.status_code == 200:
                    for toot in resp.json():
                        content = re.sub(r'<[^>]+>', '', toot.get("content", "")).strip()
                        if len(content) < 30:
                            continue
                        if language and toot.get("language") != language:
                            continue

                        account = toot.get("account", {})
                        all_posts.append(SocialPost(
                            platform="mastodon",
                            post_id=toot.get("id", ""),
                            text=content,
                            author=account.get("acct", ""),
                            url=toot.get("url", ""),
                            timestamp=toot.get("created_at", ""),
                            language=toot.get("language", "en"),
                            engagement={
                                "reblogs": toot.get("reblogs_count", 0),
                                "favourites": toot.get("favourites_count", 0),
                                "replies": toot.get("replies_count", 0),
                            },
                            metadata={
                                "instance": urlparse(instance).netloc,
                                "search_query": query,
                                "tags": [t.get("name", "") for t in toot.get("tags", [])],
                            },
                        ))

            except Exception as e:
                logger.debug(f"Mastodon search {instance}: {e}")
                continue

        logger.info(f"Mastodon search '{query}': {len(all_posts)} posts found")
        return all_posts[:limit]

    def get_trending(self, instance: str | None = None, limit: int = 20) -> list[SocialPost]:
        """Get trending posts from an instance."""
        instance = instance or self.instances[0]
        posts = []

        try:
            resp = requests.get(
                f"{instance}/api/v1/trends/statuses",
                params={"limit": min(limit, 40)},
                timeout=10,
            )
            if resp.status_code == 200:
                for toot in resp.json():
                    content = re.sub(r'<[^>]+>', '', toot.get("content", "")).strip()
                    if len(content) < 30:
                        continue
                    account = toot.get("account", {})
                    posts.append(SocialPost(
                        platform="mastodon",
                        post_id=toot.get("id", ""),
                        text=content,
                        author=account.get("acct", ""),
                        url=toot.get("url", ""),
                        timestamp=toot.get("created_at", ""),
                        language=toot.get("language", "en"),
                        engagement={
                            "reblogs": toot.get("reblogs_count", 0),
                            "favourites": toot.get("favourites_count", 0),
                        },
                        metadata={"instance": urlparse(instance).netloc, "trending": True},
                    ))
        except Exception as e:
            logger.debug(f"Mastodon trending: {e}")

        return posts

    def stream_public(
        self,
        instance: str | None = None,
        keywords: list[str] | None = None,
        duration: float = 60.0,
        callback: Callable[[SocialPost], None] | None = None,
    ) -> list[SocialPost]:
        """
        Stream public timeline in real-time using SSE.

        Args:
            instance: Mastodon instance URL
            keywords: Filter posts containing these keywords
            duration: Stream duration in seconds
            callback: Called for each matching post
        """
        instance = instance or self.instances[0]
        posts = []
        self._streaming = True

        try:
            import httpx

            start = time.time()
            with httpx.stream(
                "GET",
                f"{instance}/api/v1/streaming/public",
                timeout=duration + 5,
            ) as response:
                for line in response.iter_lines():
                    if not self._streaming or (time.time() - start) > duration:
                        break

                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:])
                            if data.get("type") != "update":
                                continue

                            payload = json.loads(data.get("payload", "{}")) if isinstance(data.get("payload"), str) else data.get("payload", {})
                            content = re.sub(r'<[^>]+>', '', payload.get("content", "")).strip()

                            if len(content) < 30:
                                continue

                            # Keyword filter
                            if keywords:
                                content_lower = content.lower()
                                if not any(kw.lower() in content_lower for kw in keywords):
                                    continue

                            account = payload.get("account", {})
                            post = SocialPost(
                                platform="mastodon",
                                post_id=payload.get("id", ""),
                                text=content,
                                author=account.get("acct", ""),
                                url=payload.get("url", ""),
                                timestamp=payload.get("created_at", ""),
                                language=payload.get("language", "en"),
                                engagement={
                                    "reblogs": payload.get("reblogs_count", 0),
                                    "favourites": payload.get("favourites_count", 0),
                                },
                                metadata={
                                    "instance": urlparse(instance).netloc,
                                    "stream": True,
                                },
                            )
                            posts.append(post)

                            if callback:
                                callback(post)

                        except (json.JSONDecodeError, KeyError):
                            continue

        except ImportError:
            logger.warning("httpx required for streaming: pip install httpx")
            # Fallback: poll every 5 seconds
            return self._poll_stream(instance, keywords, duration)
        except Exception as e:
            logger.error(f"Mastodon stream error: {e}")

        self._streaming = False
        logger.info(f"Mastodon stream: {len(posts)} posts in {time.time() - start:.1f}s")
        return posts

    def _poll_stream(
        self,
        instance: str,
        keywords: list[str] | None,
        duration: float,
    ) -> list[SocialPost]:
        """Fallback polling-based stream."""
        posts = []
        seen_ids = set()
        start = time.time()

        while (time.time() - start) < duration:
            new_posts = self.get_public_timeline(instance, limit=40)
            for post in new_posts:
                if post.post_id in seen_ids:
                    continue
                seen_ids.add(post.post_id)

                if keywords:
                    text_lower = post.text.lower()
                    if not any(kw.lower() in text_lower for kw in keywords):
                        continue
                posts.append(post)

            time.sleep(5)

        return posts

    def stop_stream(self) -> None:
        """Stop the current stream."""
        self._streaming = False


# ---------------------------------------------------------------------------
# RSS FEEDS Connector (FREE - No auth, unlimited)
# ---------------------------------------------------------------------------

class RSSConnector:
    """
    Monitor news from major outlets worldwide via RSS/Atom feeds.

    Features:
      - No API key needed
      - No rate limits
      - Multilingual (en, fr, ar, de, es, etc.)
      - Major news outlets: BBC, CNN, Al Jazeera, Fox News, Le Monde, etc.
      - Custom feed URLs supported
      - Ideal for detecting: loaded language, fear appeals, framing, etc.
    """

    # Default feeds organized by language/region
    DEFAULT_FEEDS = {
        # English
        "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "BBC Politics": "http://feeds.bbci.co.uk/news/politics/rss.xml",
        "CNN World": "http://rss.cnn.com/rss/edition_world.rss",
        "CNN Politics": "http://rss.cnn.com/rss/cnn_allpolitics.rss",
        "Fox News Politics": "https://moxie.foxnews.com/google-publisher/politics.xml",
        "Al Jazeera EN": "https://www.aljazeera.com/xml/rss/all.xml",
        "NPR News": "https://feeds.npr.org/1001/rss.xml",
        "The Guardian World": "https://www.theguardian.com/world/rss",
        # French
        "France24 FR": "https://www.france24.com/fr/rss",
        "Le Monde": "https://www.lemonde.fr/rss/une.xml",
        "RFI FR": "https://www.rfi.fr/fr/rss",
        # Arabic
        "Al Jazeera AR": "https://www.aljazeera.net/aljazeerarss/a7c186be-1baa-4bd4-9d80-a84db769f779/73d0e1b4-532f-45ef-b135-bfdff8b8cab9",
        "BBC Arabic": "http://feeds.bbci.co.uk/arabic/rss.xml",
        # German
        "DW German": "https://rss.dw.com/rdf/rss-de-all",
        "Spiegel": "https://www.spiegel.de/schlagzeilen/tops/index.rss",
        # Spanish
        "BBC Mundo": "http://feeds.bbci.co.uk/mundo/rss.xml",
        "El Pais": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    }

    # Language detection from feed URL
    FEED_LANGUAGES = {
        "France24 FR": "fr", "Le Monde": "fr", "RFI FR": "fr",
        "Al Jazeera AR": "ar", "BBC Arabic": "ar",
        "DW German": "de", "Spiegel": "de",
        "BBC Mundo": "es", "El Pais": "es",
    }

    def __init__(self, feeds: dict[str, str] | None = None) -> None:
        self.feeds = feeds or self.DEFAULT_FEEDS
        logger.info(f"RSS connector ready ({len(self.feeds)} feeds configured)")

    @property
    def is_available(self) -> bool:
        return True  # Always available

    def fetch_feed(
        self,
        feed_name: str,
        feed_url: str,
        limit: int = 20,
    ) -> list[SocialPost]:
        """Fetch articles from a single RSS feed."""
        posts = []

        try:
            feed = feedparser.parse(feed_url)

            if not feed.entries:
                return []

            lang = self.FEED_LANGUAGES.get(feed_name, "en")

            for entry in feed.entries[:limit]:
                title = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", ""))
                # Clean HTML from summary
                clean_summary = re.sub(r'<[^>]+>', '', summary).strip()

                # Combine title + summary for richer analysis
                text = f"{title}. {clean_summary}" if clean_summary else title
                if len(text) < 30:
                    continue

                # Generate stable ID
                post_id = hashlib.md5(
                    (entry.get("link", "") + title).encode()
                ).hexdigest()[:12]

                posts.append(SocialPost(
                    platform="rss",
                    post_id=post_id,
                    text=text,
                    author=feed_name,
                    url=entry.get("link", ""),
                    timestamp=entry.get("published", entry.get("updated", "")),
                    language=lang,
                    engagement={},
                    metadata={
                        "feed_name": feed_name,
                        "feed_url": feed_url,
                        "title": title,
                        "categories": [t.get("term", "") for t in entry.get("tags", [])],
                    },
                ))

        except Exception as e:
            logger.debug(f"RSS feed {feed_name} failed: {e}")

        return posts

    def fetch_all(
        self,
        languages: list[str] | None = None,
        limit_per_feed: int = 10,
    ) -> list[SocialPost]:
        """Fetch articles from all configured feeds."""
        all_posts = []

        for name, url in self.feeds.items():
            # Filter by language if specified
            if languages:
                feed_lang = self.FEED_LANGUAGES.get(name, "en")
                if feed_lang not in languages:
                    continue

            posts = self.fetch_feed(name, url, limit=limit_per_feed)
            all_posts.extend(posts)

        logger.info(f"RSS: {len(all_posts)} articles from {len(self.feeds)} feeds")
        return all_posts

    def fetch_by_language(self, language: str, limit: int = 30) -> list[SocialPost]:
        """Fetch articles in a specific language."""
        return self.fetch_all(languages=[language], limit_per_feed=limit)

    def search_in_feeds(
        self,
        query: str,
        languages: list[str] | None = None,
    ) -> list[SocialPost]:
        """Search for articles containing specific keywords."""
        all_posts = self.fetch_all(languages=languages)
        query_lower = query.lower()
        keywords = [kw.strip() for kw in query_lower.split(" OR ") if kw.strip()]

        matched = []
        for post in all_posts:
            text_lower = post.text.lower()
            if any(kw in text_lower for kw in keywords):
                matched.append(post)

        logger.info(f"RSS search '{query}': {len(matched)}/{len(all_posts)} matched")
        return matched

    def add_feed(self, name: str, url: str, language: str = "en") -> None:
        """Add a custom RSS feed."""
        self.feeds[name] = url
        if language != "en":
            self.FEED_LANGUAGES[name] = language
        logger.info(f"Added RSS feed: {name} ({url})")

    def get_feed_list(self) -> list[dict]:
        """List all configured feeds."""
        return [
            {
                "name": name,
                "url": url,
                "language": self.FEED_LANGUAGES.get(name, "en"),
            }
            for name, url in self.feeds.items()
        ]


# ---------------------------------------------------------------------------
# YOUTUBE TRANSCRIPTS Connector (FREE - No API key needed)
# ---------------------------------------------------------------------------

class YouTubeTranscriptConnector:
    """
    Analyze YouTube video transcripts without any API key.

    Uses youtube-transcript-api (scraping-based, no official API needed).

    Features:
      - No API key required
      - Auto-generated + manual subtitles
      - Multi-language support (50+ languages)
      - Full video transcripts for persuasion analysis
      - Timestamped segments
    """

    def __init__(self) -> None:
        self._api = None
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            self._api = YouTubeTranscriptApi()
            logger.info("YouTube Transcript connector ready (no API key needed)")
        except ImportError:
            logger.warning("youtube-transcript-api not installed: pip install youtube-transcript-api")

    @property
    def is_available(self) -> bool:
        return self._api is not None

    def get_transcript(
        self,
        video_id: str,
        languages: list[str] | None = None,
    ) -> list[SocialPost]:
        """
        Get transcript of a YouTube video as analyzable posts.

        Splits transcript into chunks (~60s each) for granular analysis.
        """
        if not self._api:
            return []

        languages = languages or ["en", "fr", "ar", "es", "de"]

        try:
            transcript = self._api.fetch(video_id, languages=languages)
            segments = list(transcript)

            if not segments:
                return []

            # Group segments into ~60-second chunks
            posts = []
            chunk_text = ""
            chunk_start = 0
            chunk_segments = 0

            for seg in segments:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue

                if chunk_segments == 0:
                    chunk_start = seg.start

                chunk_text += " " + seg_text
                chunk_segments += 1

                # Create a post every ~60 seconds or 500 chars
                if seg.start - chunk_start >= 60 or len(chunk_text) > 500:
                    if len(chunk_text.strip()) > 30:
                        post_id = f"{video_id}_{int(chunk_start)}"
                        posts.append(SocialPost(
                            platform="youtube_transcript",
                            post_id=post_id,
                            text=chunk_text.strip(),
                            author="",
                            url=f"https://youtube.com/watch?v={video_id}&t={int(chunk_start)}",
                            timestamp=f"{int(chunk_start)}s - {int(seg.start)}s",
                            language=languages[0] if languages else "en",
                            engagement={},
                            metadata={
                                "video_id": video_id,
                                "start_time": round(chunk_start, 1),
                                "end_time": round(seg.start, 1),
                                "num_segments": chunk_segments,
                            },
                        ))

                    chunk_text = ""
                    chunk_segments = 0

            # Don't forget last chunk
            if chunk_text.strip() and len(chunk_text.strip()) > 30:
                post_id = f"{video_id}_{int(chunk_start)}"
                posts.append(SocialPost(
                    platform="youtube_transcript",
                    post_id=post_id,
                    text=chunk_text.strip(),
                    author="",
                    url=f"https://youtube.com/watch?v={video_id}&t={int(chunk_start)}",
                    timestamp=f"{int(chunk_start)}s+",
                    language=languages[0] if languages else "en",
                    metadata={"video_id": video_id, "start_time": round(chunk_start, 1)},
                ))

            logger.info(f"YouTube transcript {video_id}: {len(posts)} chunks from {len(segments)} segments")
            return posts

        except Exception as e:
            logger.error(f"YouTube transcript {video_id} failed: {e}")
            return []

    def get_available_languages(self, video_id: str) -> list[str]:
        """List available transcript languages for a video."""
        if not self._api:
            return []

        try:
            transcript_list = self._api.list(video_id)
            return [t.language_code for t in transcript_list]
        except Exception as e:
            logger.debug(f"Language list failed: {e}")
            return []

    def search_and_analyze(
        self,
        video_ids: list[str],
        languages: list[str] | None = None,
    ) -> list[SocialPost]:
        """Fetch transcripts from multiple videos."""
        all_posts = []
        for vid in video_ids:
            posts = self.get_transcript(vid, languages)
            all_posts.extend(posts)
            time.sleep(0.5)  # Be polite
        return all_posts


# ---------------------------------------------------------------------------
# HACKER NEWS Connector (FREE - No auth, no limits)
# ---------------------------------------------------------------------------

class HackerNewsConnector:
    """
    Monitor Hacker News via Firebase API.

    Features:
      - Completely free, no auth
      - No rate limits
      - Real-time via Server-Sent Events
      - Good for tech/society debate analysis
    """

    BASE_URL = "https://hacker-news.firebaseio.com/v0"

    def __init__(self) -> None:
        logger.info("HackerNews connector ready (no auth needed)")

    @property
    def is_available(self) -> bool:
        return True

    def get_top_stories(self, limit: int = 20, include_comments: bool = True) -> list[SocialPost]:
        """Fetch top stories and their comments."""
        posts = []

        try:
            resp = requests.get(f"{self.BASE_URL}/topstories.json", timeout=10)
            if resp.status_code != 200:
                return []

            story_ids = resp.json()[:limit]

            for sid in story_ids:
                try:
                    item_resp = requests.get(f"{self.BASE_URL}/item/{sid}.json", timeout=5)
                    if item_resp.status_code != 200:
                        continue

                    item = item_resp.json()
                    if not item:
                        continue

                    title = item.get("title", "")

                    # Fetch top comments for richer text
                    comment_texts = []
                    if include_comments:
                        kids = item.get("kids", [])[:3]
                        for kid_id in kids:
                            try:
                                c_resp = requests.get(f"{self.BASE_URL}/item/{kid_id}.json", timeout=5)
                                if c_resp.status_code == 200:
                                    c_data = c_resp.json()
                                    c_text = c_data.get("text", "")
                                    if c_text:
                                        clean = re.sub(r'<[^>]+>', '', c_text).strip()
                                        if len(clean) > 20:
                                            comment_texts.append(clean[:300])
                            except Exception:
                                continue

                    # Title as one post
                    if title and len(title) > 20:
                        posts.append(SocialPost(
                            platform="hackernews",
                            post_id=str(sid),
                            text=title,
                            author=item.get("by", ""),
                            url=item.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                            timestamp=datetime.fromtimestamp(item.get("time", 0)).isoformat(),
                            language="en",
                            engagement={
                                "score": item.get("score", 0),
                                "num_comments": item.get("descendants", 0),
                            },
                            metadata={"type": "story", "hn_url": f"https://news.ycombinator.com/item?id={sid}"},
                        ))

                    # Comments as separate posts (richer text for analysis)
                    for i, comment in enumerate(comment_texts):
                        posts.append(SocialPost(
                            platform="hackernews",
                            post_id=f"{sid}_c{i}",
                            text=comment,
                            author="",
                            url=f"https://news.ycombinator.com/item?id={sid}",
                            timestamp="",
                            language="en",
                            engagement={},
                            metadata={"type": "comment", "parent_story": title},
                        ))

                except Exception as e:
                    logger.debug(f"HN item {sid}: {e}")
                    continue

            logger.info(f"HackerNews: {len(posts)} items fetched")

        except Exception as e:
            logger.error(f"HackerNews failed: {e}")

        return posts

    def get_new_stories(self, limit: int = 20) -> list[SocialPost]:
        """Fetch newest stories."""
        posts = []
        try:
            resp = requests.get(f"{self.BASE_URL}/newstories.json", timeout=10)
            if resp.status_code == 200:
                story_ids = resp.json()[:limit]
                for sid in story_ids:
                    item_resp = requests.get(f"{self.BASE_URL}/item/{sid}.json", timeout=5)
                    if item_resp.status_code == 200:
                        item = item_resp.json()
                        if item and item.get("title") and len(item["title"]) > 20:
                            posts.append(SocialPost(
                                platform="hackernews",
                                post_id=str(sid),
                                text=item["title"],
                                author=item.get("by", ""),
                                url=item.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                                timestamp=datetime.fromtimestamp(item.get("time", 0)).isoformat(),
                                language="en",
                                engagement={"score": item.get("score", 0)},
                                metadata={"type": "story"},
                            ))
        except Exception as e:
            logger.debug(f"HN new stories: {e}")

        return posts


# ---------------------------------------------------------------------------
# Twitter/X Connector (PAID - requires Bearer Token)
# ---------------------------------------------------------------------------

class TwitterConnector:
    """Connect to Twitter/X API v2 for searching and streaming tweets."""

    BASE_URL = "https://api.twitter.com/2"

    def __init__(self) -> None:
        self.bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
        if not self.bearer_token:
            logger.info("TWITTER_BEARER_TOKEN not set — Twitter integration disabled (paid API)")

    @property
    def is_available(self) -> bool:
        return bool(self.bearer_token)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def search(
        self,
        query: str,
        max_results: int = 50,
        language: str = "en",
    ) -> list[SocialPost]:
        """Search recent tweets matching a query."""
        if not self.is_available:
            return []

        try:
            params = {
                "query": f"{query} lang:{language} -is:retweet",
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,author_id,public_metrics,lang",
                "expansions": "author_id",
                "user.fields": "username,name",
            }
            response = requests.get(
                f"{self.BASE_URL}/tweets/search/recent",
                headers=self._headers(),
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            authors = {}
            for user in data.get("includes", {}).get("users", []):
                authors[user["id"]] = user.get("username", "")

            posts = []
            for tweet in data.get("data", []):
                metrics = tweet.get("public_metrics", {})
                posts.append(SocialPost(
                    platform="twitter",
                    post_id=tweet["id"],
                    text=tweet["text"],
                    author=authors.get(tweet.get("author_id", ""), ""),
                    url=f"https://twitter.com/i/status/{tweet['id']}",
                    timestamp=tweet.get("created_at", ""),
                    language=tweet.get("lang", language),
                    engagement={
                        "likes": metrics.get("like_count", 0),
                        "retweets": metrics.get("retweet_count", 0),
                        "replies": metrics.get("reply_count", 0),
                    },
                ))

            logger.info(f"Twitter search '{query}': {len(posts)} tweets found")
            return posts

        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Unified Social Monitor
# ---------------------------------------------------------------------------

class SocialMonitor:
    """
    Unified social media monitor for PersuasiX.

    Fetches content from multiple platforms and runs persuasion analysis.

    FREE sources (no API keys):
      - Mastodon (real-time streaming)
      - RSS Feeds (news worldwide)
      - YouTube Transcripts (video content)
      - Hacker News (debates)

    PAID sources (optional):
      - Twitter/X (bearer token)
    """

    def __init__(self, pipeline: Any = None) -> None:
        # FREE connectors
        self.mastodon = MastodonConnector()
        self.rss = RSSConnector()
        self.youtube_transcripts = YouTubeTranscriptConnector()
        self.hackernews = HackerNewsConnector()

        # PAID connectors (optional)
        self.twitter = TwitterConnector()

        self._pipeline = pipeline
        logger.info("SocialMonitor initialized (free sources: mastodon, rss, youtube, hackernews)")

    def _get_pipeline(self):
        if self._pipeline is None:
            from src.pipeline.persuasix_pipeline import PersuasixPipeline
            self._pipeline = PersuasixPipeline.from_default_models(device="cpu")
        return self._pipeline

    def _analyze_post(self, post: SocialPost) -> SocialAnalysisResult:
        """Analyze a single social media post."""
        pipe = self._get_pipeline()
        text = post.text[:3000]
        result = pipe.analyze(text, language=post.language)
        return SocialAnalysisResult(
            post=post,
            is_persuasive=result.is_persuasive,
            manipulation_score=result.manipulation_score,
            severity_score=result.severity_score,
            techniques=result.techniques,
            highlighted_phrases=result.highlighted_phrases,
            explanation=result.explanation,
        )

    def _analyze_posts(self, posts: list[SocialPost]) -> list[SocialAnalysisResult]:
        """Analyze multiple posts."""
        results = []
        for post in posts:
            try:
                result = self._analyze_post(post)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze post {post.post_id}: {e}")
        return results

    # ===== FREE APIs =====

    def search_mastodon(
        self,
        query: str,
        limit: int = 30,
        language: str | None = None,
        analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Search Mastodon (FREE, no auth)."""
        posts = self.mastodon.search(query, limit, language)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def stream_mastodon(
        self,
        keywords: list[str] | None = None,
        duration: float = 60.0,
        instance: str | None = None,
        analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Stream Mastodon public timeline in real-time (FREE)."""
        posts = self.mastodon.stream_public(instance, keywords, duration)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def get_mastodon_trending(
        self,
        instance: str | None = None,
        analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Get trending Mastodon posts (FREE)."""
        posts = self.mastodon.get_trending(instance)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def fetch_rss_news(
        self,
        languages: list[str] | None = None,
        limit_per_feed: int = 10,
        analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Fetch news from RSS feeds (FREE, unlimited)."""
        posts = self.rss.fetch_all(languages, limit_per_feed)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def search_rss(
        self,
        query: str,
        languages: list[str] | None = None,
        analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Search in RSS feeds for specific topics (FREE)."""
        posts = self.rss.search_in_feeds(query, languages)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def analyze_youtube_video(
        self,
        video_id: str,
        languages: list[str] | None = None,
        analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Analyze a YouTube video transcript (FREE, no API key)."""
        posts = self.youtube_transcripts.get_transcript(video_id, languages)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def fetch_hackernews(
        self,
        limit: int = 20,
        include_comments: bool = True,
        analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Fetch Hacker News stories and comments (FREE)."""
        posts = self.hackernews.get_top_stories(limit, include_comments)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    # ===== PAID APIs (optional) =====

    def search_twitter(
        self, query: str, max_results: int = 50, language: str = "en", analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Search Twitter (PAID - requires bearer token)."""
        posts = self.twitter.search(query, max_results, language)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    # ===== MULTI-PLATFORM =====

    def multi_platform_search(
        self,
        query: str,
        platforms: list[str] | None = None,
        limit: int = 20,
        analyze: bool = True,
    ) -> dict[str, list]:
        """
        Search across multiple platforms simultaneously.

        Default platforms: mastodon, rss, hackernews (all free).
        """
        platforms = platforms or ["mastodon", "rss", "hackernews"]
        results: dict[str, list] = {}

        if "mastodon" in platforms:
            posts = self.mastodon.search(query, limit)
            results["mastodon"] = self._analyze_posts(posts) if analyze else posts

        if "rss" in platforms:
            posts = self.rss.search_in_feeds(query)
            results["rss"] = self._analyze_posts(posts) if analyze else posts

        if "hackernews" in platforms:
            posts = self.hackernews.get_top_stories(limit, include_comments=True)
            # Filter by query
            query_lower = query.lower()
            filtered = [p for p in posts if query_lower in p.text.lower()]
            results["hackernews"] = self._analyze_posts(filtered) if analyze else filtered

        if "twitter" in platforms and self.twitter.is_available:
            posts = self.twitter.search(query, limit)
            results["twitter"] = self._analyze_posts(posts) if analyze else posts

        if "youtube" in platforms and self.youtube_transcripts.is_available:
            results["youtube"] = []  # Requires video IDs, not keyword search

        total = sum(len(v) for v in results.values())
        logger.info(f"Multi-platform search '{query}': {total} results across {len(results)} platforms")
        return results

    def monitor_realtime(
        self,
        keywords: list[str],
        duration: float = 120.0,
        platforms: list[str] | None = None,
        callback: Callable[[SocialAnalysisResult], None] | None = None,
    ) -> list[SocialAnalysisResult]:
        """
        Real-time monitoring across platforms.

        Streams from Mastodon + polls RSS every 30s.
        """
        platforms = platforms or ["mastodon", "rss"]
        all_results: list[SocialAnalysisResult] = []
        start_time = time.time()

        def _mastodon_worker():
            posts = self.mastodon.stream_public(
                keywords=keywords,
                duration=duration,
            )
            for post in posts:
                try:
                    result = self._analyze_post(post)
                    all_results.append(result)
                    if callback:
                        callback(result)
                except Exception:
                    continue

        def _rss_worker():
            seen_ids = set()
            while (time.time() - start_time) < duration:
                posts = self.rss.search_in_feeds(" OR ".join(keywords))
                for post in posts:
                    if post.post_id in seen_ids:
                        continue
                    seen_ids.add(post.post_id)
                    try:
                        result = self._analyze_post(post)
                        all_results.append(result)
                        if callback:
                            callback(result)
                    except Exception:
                        continue
                time.sleep(30)  # Poll RSS every 30s

        threads = []
        if "mastodon" in platforms:
            t = threading.Thread(target=_mastodon_worker, daemon=True)
            threads.append(t)
            t.start()

        if "rss" in platforms:
            t = threading.Thread(target=_rss_worker, daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=duration + 10)

        logger.info(f"Real-time monitoring: {len(all_results)} results in {time.time() - start_time:.1f}s")
        return all_results

    # ===== REPORTING =====

    def get_platform_status(self) -> dict[str, dict]:
        """Check which platforms are available."""
        return {
            "mastodon": {"available": True, "type": "free", "realtime": True},
            "rss": {"available": True, "type": "free", "realtime": False},
            "youtube_transcripts": {"available": self.youtube_transcripts.is_available, "type": "free", "realtime": False},
            "hackernews": {"available": True, "type": "free", "realtime": True},
            "twitter": {"available": self.twitter.is_available, "type": "paid", "realtime": True},
        }

    def generate_report(self, results: list[SocialAnalysisResult]) -> dict:
        """Generate a summary report from analysis results."""
        if not results:
            return {"total": 0}

        total = len(results)
        manipulative = [r for r in results if r.is_persuasive]
        avg_score = sum(r.manipulation_score for r in results) / total

        technique_freq: dict[str, int] = {}
        for r in results:
            for t in r.techniques:
                technique_freq[t] = technique_freq.get(t, 0) + 1

        top_manipulative = sorted(manipulative, key=lambda r: -r.manipulation_score)[:5]

        return {
            "total_posts": total,
            "manipulative_count": len(manipulative),
            "neutral_count": total - len(manipulative),
            "manipulation_rate": round(len(manipulative) / total * 100, 1),
            "avg_manipulation_score": round(avg_score * 100, 1),
            "technique_frequency": dict(sorted(technique_freq.items(), key=lambda x: -x[1])),
            "top_manipulative": [r.to_dict() for r in top_manipulative],
            "platform_breakdown": {
                platform: len([r for r in results if r.post.platform == platform])
                for platform in set(r.post.platform for r in results)
            },
            "language_breakdown": {
                lang: len([r for r in results if r.post.language == lang])
                for lang in set(r.post.language for r in results)
            },
        }
