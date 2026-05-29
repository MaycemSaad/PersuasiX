"""
Social media integration for real-time persuasion monitoring.

Supports:
  - Twitter/X API v2 (stream + search)
  - Reddit API (PRAW)
  - YouTube comments (via Data API v3)
  - Telegram channels (via Telethon)

Each connector:
  1. Fetches posts/comments matching keywords or from specific sources
  2. Runs PersuasiX analysis on each
  3. Stores results + triggers alerts for high-score content

Usage:
    from src.pipeline.social_monitor import SocialMonitor
    monitor = SocialMonitor()
    monitor.search_twitter("climate change", max_results=50)
    monitor.monitor_subreddit("politics", limit=25)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator

import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SocialPost:
    """A single social media post/comment."""
    platform: str           # twitter, reddit, youtube, telegram
    post_id: str
    text: str
    author: str = ""
    url: str = ""
    timestamp: str = ""
    language: str = "en"
    engagement: dict = field(default_factory=dict)  # likes, retweets, etc.
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "post_id": self.post_id,
            "text": self.text,
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
# Twitter/X Connector
# ---------------------------------------------------------------------------

class TwitterConnector:
    """Connect to Twitter/X API v2 for searching and streaming tweets."""

    BASE_URL = "https://api.twitter.com/2"

    def __init__(self) -> None:
        self.bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
        if not self.bearer_token:
            logger.info("TWITTER_BEARER_TOKEN not set — Twitter integration disabled")

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

            # Build author lookup
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
                        "impressions": metrics.get("impression_count", 0),
                    },
                ))

            logger.info(f"Twitter search '{query}': {len(posts)} tweets found")
            return posts

        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
            return []

    def get_user_tweets(
        self,
        username: str,
        max_results: int = 20,
    ) -> list[SocialPost]:
        """Get recent tweets from a specific user."""
        if not self.is_available:
            return []

        try:
            # First get user ID
            user_resp = requests.get(
                f"{self.BASE_URL}/users/by/username/{username}",
                headers=self._headers(),
                timeout=10,
            )
            user_resp.raise_for_status()
            user_id = user_resp.json()["data"]["id"]

            # Then get tweets
            params = {
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,public_metrics,lang",
                "exclude": "retweets,replies",
            }
            resp = requests.get(
                f"{self.BASE_URL}/users/{user_id}/tweets",
                headers=self._headers(),
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            posts = []
            for tweet in data.get("data", []):
                posts.append(SocialPost(
                    platform="twitter",
                    post_id=tweet["id"],
                    text=tweet["text"],
                    author=username,
                    url=f"https://twitter.com/{username}/status/{tweet['id']}",
                    timestamp=tweet.get("created_at", ""),
                    language=tweet.get("lang", "en"),
                ))

            return posts

        except Exception as e:
            logger.error(f"Twitter user fetch failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Reddit Connector
# ---------------------------------------------------------------------------

class RedditConnector:
    """Connect to Reddit API for monitoring subreddits and posts."""

    BASE_URL = "https://oauth.reddit.com"
    AUTH_URL = "https://www.reddit.com/api/v1/access_token"

    def __init__(self) -> None:
        self.client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        self.client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        self.user_agent = os.environ.get("REDDIT_USER_AGENT", "PersuasiX/1.0")
        self._token: str | None = None
        self._token_expires: float = 0

        if not self.client_id or not self.client_secret:
            logger.info("Reddit API credentials not set — Reddit integration disabled")

    @property
    def is_available(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _authenticate(self) -> None:
        """Get OAuth2 token."""
        if self._token and time.time() < self._token_expires:
            return

        try:
            resp = requests.post(
                self.AUTH_URL,
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._token_expires = time.time() + data.get("expires_in", 3600) - 60
        except Exception as e:
            logger.error(f"Reddit authentication failed: {e}")

    def _headers(self) -> dict:
        self._authenticate()
        return {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": self.user_agent,
        }

    def search_subreddit(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 25,
    ) -> list[SocialPost]:
        """Fetch posts from a subreddit."""
        if not self.is_available:
            return []

        try:
            resp = requests.get(
                f"{self.BASE_URL}/r/{subreddit}/{sort}",
                headers=self._headers(),
                params={"limit": min(limit, 100)},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            posts = []
            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})
                text = post_data.get("selftext", "") or post_data.get("title", "")
                if len(text.strip()) < 20:
                    text = f"{post_data.get('title', '')}. {post_data.get('selftext', '')}"

                if len(text.strip()) < 20:
                    continue

                posts.append(SocialPost(
                    platform="reddit",
                    post_id=post_data.get("id", ""),
                    text=text.strip(),
                    author=post_data.get("author", ""),
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    timestamp=datetime.fromtimestamp(
                        post_data.get("created_utc", 0)
                    ).isoformat() if post_data.get("created_utc") else "",
                    engagement={
                        "score": post_data.get("score", 0),
                        "upvote_ratio": post_data.get("upvote_ratio", 0),
                        "num_comments": post_data.get("num_comments", 0),
                        "awards": post_data.get("total_awards_received", 0),
                    },
                    metadata={
                        "subreddit": subreddit,
                        "flair": post_data.get("link_flair_text", ""),
                        "is_self": post_data.get("is_self", False),
                    },
                ))

            logger.info(f"Reddit r/{subreddit}: {len(posts)} posts fetched")
            return posts

        except Exception as e:
            logger.error(f"Reddit fetch failed: {e}")
            return []

    def search(self, query: str, limit: int = 25) -> list[SocialPost]:
        """Search Reddit for posts matching a query."""
        if not self.is_available:
            return []

        try:
            resp = requests.get(
                f"{self.BASE_URL}/search",
                headers=self._headers(),
                params={"q": query, "limit": min(limit, 100), "sort": "relevance", "t": "week"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            posts = []
            for child in data.get("data", {}).get("children", []):
                pd = child.get("data", {})
                text = f"{pd.get('title', '')}. {pd.get('selftext', '')}".strip()
                if len(text) < 20:
                    continue

                posts.append(SocialPost(
                    platform="reddit",
                    post_id=pd.get("id", ""),
                    text=text,
                    author=pd.get("author", ""),
                    url=f"https://reddit.com{pd.get('permalink', '')}",
                    engagement={"score": pd.get("score", 0), "num_comments": pd.get("num_comments", 0)},
                    metadata={"subreddit": pd.get("subreddit", "")},
                ))

            return posts

        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
            return []


# ---------------------------------------------------------------------------
# YouTube Connector
# ---------------------------------------------------------------------------

class YouTubeConnector:
    """Fetch and analyze YouTube video comments via Data API v3."""

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(self) -> None:
        self.api_key = os.environ.get("YOUTUBE_API_KEY", "")
        if not self.api_key:
            logger.info("YOUTUBE_API_KEY not set — YouTube integration disabled")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_comments(self, video_id: str, max_results: int = 50) -> list[SocialPost]:
        """Fetch top-level comments from a YouTube video."""
        if not self.is_available:
            return []

        try:
            params = {
                "key": self.api_key,
                "videoId": video_id,
                "part": "snippet",
                "maxResults": min(max_results, 100),
                "order": "relevance",
                "textFormat": "plainText",
            }
            resp = requests.get(
                f"{self.BASE_URL}/commentThreads",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            posts = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                text = snippet.get("textDisplay", "")
                if len(text.strip()) < 10:
                    continue

                posts.append(SocialPost(
                    platform="youtube",
                    post_id=item.get("id", ""),
                    text=text,
                    author=snippet.get("authorDisplayName", ""),
                    url=f"https://youtube.com/watch?v={video_id}&lc={item.get('id', '')}",
                    timestamp=snippet.get("publishedAt", ""),
                    engagement={
                        "likes": snippet.get("likeCount", 0),
                    },
                    metadata={"video_id": video_id},
                ))

            logger.info(f"YouTube video {video_id}: {len(posts)} comments fetched")
            return posts

        except Exception as e:
            logger.error(f"YouTube fetch failed: {e}")
            return []

    def search_videos(self, query: str, max_results: int = 10) -> list[dict]:
        """Search YouTube videos by query (returns video metadata, not comments)."""
        if not self.is_available:
            return []

        try:
            params = {
                "key": self.api_key,
                "q": query,
                "part": "snippet",
                "type": "video",
                "maxResults": min(max_results, 50),
                "order": "relevance",
            }
            resp = requests.get(f"{self.BASE_URL}/search", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            videos = []
            for item in data.get("items", []):
                vid_id = item.get("id", {}).get("videoId", "")
                snippet = item.get("snippet", {})
                videos.append({
                    "video_id": vid_id,
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "url": f"https://youtube.com/watch?v={vid_id}",
                })

            return videos

        except Exception as e:
            logger.error(f"YouTube search failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Unified Social Monitor
# ---------------------------------------------------------------------------

class SocialMonitor:
    """
    Unified social media monitor that fetches content from multiple
    platforms and runs PersuasiX analysis on each post.
    """

    def __init__(self, pipeline: Any = None) -> None:
        self.twitter = TwitterConnector()
        self.reddit = RedditConnector()
        self.youtube = YouTubeConnector()
        self._pipeline = pipeline

    def _get_pipeline(self):
        if self._pipeline is None:
            from src.pipeline.persuasix_pipeline import PersuasixPipeline
            self._pipeline = PersuasixPipeline.from_default_models(device="cpu")
        return self._pipeline

    def _analyze_post(self, post: SocialPost) -> SocialAnalysisResult:
        """Analyze a single social media post."""
        pipe = self._get_pipeline()
        text = post.text[:3000]  # Truncate long posts
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

    # ---- Public API ----

    def search_twitter(
        self, query: str, max_results: int = 50, language: str = "en", analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Search Twitter and optionally analyze results."""
        posts = self.twitter.search(query, max_results, language)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def monitor_subreddit(
        self, subreddit: str, sort: str = "hot", limit: int = 25, analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Monitor a Reddit subreddit."""
        posts = self.reddit.search_subreddit(subreddit, sort, limit)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def search_reddit(
        self, query: str, limit: int = 25, analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Search Reddit posts."""
        posts = self.reddit.search(query, limit)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def analyze_youtube_comments(
        self, video_id: str, max_results: int = 50, analyze: bool = True,
    ) -> list[SocialAnalysisResult] | list[SocialPost]:
        """Fetch and analyze YouTube comments."""
        posts = self.youtube.get_comments(video_id, max_results)
        if analyze and posts:
            return self._analyze_posts(posts)
        return posts

    def multi_platform_search(
        self, query: str, platforms: list[str] | None = None, limit: int = 20,
    ) -> dict[str, list[SocialAnalysisResult]]:
        """Search across multiple platforms simultaneously."""
        platforms = platforms or ["twitter", "reddit"]
        results: dict[str, list[SocialAnalysisResult]] = {}

        if "twitter" in platforms and self.twitter.is_available:
            results["twitter"] = self.search_twitter(query, limit)

        if "reddit" in platforms and self.reddit.is_available:
            results["reddit"] = self.search_reddit(query, limit)

        return results

    def get_platform_status(self) -> dict[str, bool]:
        """Check which platforms are available."""
        return {
            "twitter": self.twitter.is_available,
            "reddit": self.reddit.is_available,
            "youtube": self.youtube.is_available,
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

        top_manipulative = sorted(
            manipulative,
            key=lambda r: -r.manipulation_score,
        )[:5]

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
        }
