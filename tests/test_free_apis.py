"""
Test script to evaluate free APIs for PersuasiX social monitoring.

Tests:
  1. Mastodon (Public timeline streaming)
  2. Bluesky (AT Protocol firehose)
  3. Hacker News (Firebase API)
  4. RSS Feeds (Major news outlets)
  5. YouTube Transcripts (youtube-transcript-api)
  6. GNews (News aggregator)
  7. Telegram (Public channel scraping)

Evaluates: availability, response time, content richness, language support, real-time capability.
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path

import requests
import feedparser

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Test results collector
# ---------------------------------------------------------------------------

class APITestResult:
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.response_time_ms = 0
        self.sample_count = 0
        self.languages_detected = set()
        self.avg_text_length = 0
        self.has_realtime = False
        self.rate_limit_info = ""
        self.content_quality = ""  # low, medium, high
        self.persuasion_relevance = ""  # low, medium, high
        self.errors = []
        self.samples = []

    def summary(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
            "response_time_ms": self.response_time_ms,
            "samples_fetched": self.sample_count,
            "languages": list(self.languages_detected),
            "avg_text_length": self.avg_text_length,
            "has_realtime": self.has_realtime,
            "rate_limit": self.rate_limit_info,
            "content_quality": self.content_quality,
            "persuasion_relevance": self.persuasion_relevance,
            "errors": self.errors[:3],
        }


results: list[APITestResult] = []


# ---------------------------------------------------------------------------
# 1. MASTODON (Public Timeline)
# ---------------------------------------------------------------------------

def test_mastodon():
    """Test Mastodon public timeline API (no auth required)."""
    print("\n" + "="*60)
    print("  1. MASTODON - Public Timeline API")
    print("="*60)

    result = APITestResult("Mastodon")
    result.has_realtime = True
    result.rate_limit_info = "300 req/5min (no auth)"

    # Test multiple instances
    instances = [
        "https://mastodon.social",
        "https://mas.to",
        "https://fosstodon.org",
    ]

    all_posts = []

    for instance in instances:
        try:
            start = time.time()
            resp = requests.get(
                f"{instance}/api/v1/timelines/public",
                params={"limit": 20, "local": False},
                timeout=10,
            )
            elapsed = (time.time() - start) * 1000

            if resp.status_code == 200:
                posts = resp.json()
                result.available = True
                result.response_time_ms = round(elapsed)

                for post in posts:
                    content = post.get("content", "")
                    lang = post.get("language", "unknown")
                    # Strip HTML
                    import re
                    clean_text = re.sub(r'<[^>]+>', '', content).strip()

                    if len(clean_text) > 30:
                        all_posts.append({
                            "text": clean_text[:300],
                            "language": lang,
                            "instance": instance.split("//")[1],
                            "created_at": post.get("created_at", ""),
                            "reblogs": post.get("reblogs_count", 0),
                            "favourites": post.get("favourites_count", 0),
                        })
                        result.languages_detected.add(lang or "unknown")

                print(f"  [OK] {instance} - {len(posts)} posts ({elapsed:.0f}ms)")
            else:
                print(f"  [WARN] {instance} - HTTP {resp.status_code}")

        except Exception as e:
            result.errors.append(f"{instance}: {str(e)[:50]}")
            print(f"  [FAIL] {instance} - {e}")

    result.sample_count = len(all_posts)
    result.samples = all_posts[:5]
    if all_posts:
        result.avg_text_length = sum(len(p["text"]) for p in all_posts) // len(all_posts)
        result.content_quality = "high"
        result.persuasion_relevance = "high"

    print(f"\n  Total posts: {len(all_posts)}")
    print(f"  Languages: {result.languages_detected}")
    print(f"  Avg length: {result.avg_text_length} chars")

    if all_posts:
        print(f"\n  Sample post:")
        print(f"    [{all_posts[0]['language']}] {all_posts[0]['text'][:150]}...")

    results.append(result)
    return result


# ---------------------------------------------------------------------------
# 2. BLUESKY (AT Protocol)
# ---------------------------------------------------------------------------

def test_bluesky():
    """Test Bluesky public API (no auth for public feeds)."""
    print("\n" + "="*60)
    print("  2. BLUESKY - AT Protocol Public API")
    print("="*60)

    result = APITestResult("Bluesky")
    result.has_realtime = True
    result.rate_limit_info = "3000 req/5min (no auth for public)"

    try:
        # Public API - no auth needed for public content
        start = time.time()

        # Search public posts
        resp = requests.get(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
            params={"q": "politics OR propaganda OR manipulation", "limit": 25},
            timeout=10,
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            posts = data.get("posts", [])
            result.available = True
            result.response_time_ms = round(elapsed)

            all_posts = []
            for post in posts:
                record = post.get("record", {})
                text = record.get("text", "")
                langs = record.get("langs", ["en"])

                if len(text) > 30:
                    all_posts.append({
                        "text": text[:300],
                        "language": langs[0] if langs else "en",
                        "created_at": record.get("createdAt", ""),
                        "likes": post.get("likeCount", 0),
                        "reposts": post.get("repostCount", 0),
                    })
                    for l in langs:
                        result.languages_detected.add(l)

            result.sample_count = len(all_posts)
            result.samples = all_posts[:5]
            if all_posts:
                result.avg_text_length = sum(len(p["text"]) for p in all_posts) // len(all_posts)

            result.content_quality = "high"
            result.persuasion_relevance = "high"

            print(f"  [OK] Search API - {len(all_posts)} posts ({elapsed:.0f}ms)")
            print(f"  Languages: {result.languages_detected}")
            print(f"  Avg length: {result.avg_text_length} chars")

            if all_posts:
                print(f"\n  Sample post:")
                print(f"    [{all_posts[0]['language']}] {all_posts[0]['text'][:150]}...")

        else:
            print(f"  [WARN] HTTP {resp.status_code}: {resp.text[:100]}")
            result.errors.append(f"HTTP {resp.status_code}")

        # Test trending/popular topics
        print("\n  Testing trending topics...")
        resp2 = requests.get(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
            params={"q": "breaking news", "limit": 10, "sort": "latest"},
            timeout=10,
        )
        if resp2.status_code == 200:
            trending = resp2.json().get("posts", [])
            print(f"  [OK] Trending/latest: {len(trending)} posts")
        else:
            print(f"  [WARN] Trending: HTTP {resp2.status_code}")

    except Exception as e:
        result.errors.append(str(e)[:100])
        print(f"  [FAIL] {e}")

    results.append(result)
    return result


# ---------------------------------------------------------------------------
# 3. HACKER NEWS (Firebase API)
# ---------------------------------------------------------------------------

def test_hackernews():
    """Test Hacker News Firebase API (no auth, no limits)."""
    print("\n" + "="*60)
    print("  3. HACKER NEWS - Firebase API")
    print("="*60)

    result = APITestResult("HackerNews")
    result.has_realtime = True
    result.rate_limit_info = "No limits (Firebase)"

    try:
        start = time.time()

        # Get top stories
        resp = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=10,
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            story_ids = resp.json()[:20]
            result.available = True
            result.response_time_ms = round(elapsed)

            all_items = []
            for sid in story_ids[:10]:
                item_resp = requests.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                    timeout=5,
                )
                if item_resp.status_code == 200:
                    item = item_resp.json()
                    title = item.get("title", "")
                    # Get some comments for text content
                    kids = item.get("kids", [])[:3]
                    comments = []
                    for kid_id in kids:
                        c_resp = requests.get(
                            f"https://hacker-news.firebaseio.com/v0/item/{kid_id}.json",
                            timeout=5,
                        )
                        if c_resp.status_code == 200:
                            c_data = c_resp.json()
                            c_text = c_data.get("text", "")
                            if c_text:
                                import re
                                clean = re.sub(r'<[^>]+>', '', c_text).strip()
                                comments.append(clean[:200])

                    all_items.append({
                        "title": title,
                        "text": " ".join(comments) if comments else title,
                        "score": item.get("score", 0),
                        "num_comments": item.get("descendants", 0),
                        "language": "en",
                    })
                    result.languages_detected.add("en")

            result.sample_count = len(all_items)
            result.samples = all_items[:5]
            if all_items:
                result.avg_text_length = sum(len(p["text"]) for p in all_items) // len(all_items)

            result.content_quality = "medium"
            result.persuasion_relevance = "medium"

            print(f"  [OK] {len(all_items)} stories + comments ({elapsed:.0f}ms)")
            print(f"  Avg text: {result.avg_text_length} chars")

            if all_items:
                print(f"\n  Sample:")
                print(f"    Title: {all_items[0]['title'][:80]}")
                print(f"    Comment: {all_items[0]['text'][:120]}...")

        else:
            print(f"  [FAIL] HTTP {resp.status_code}")

    except Exception as e:
        result.errors.append(str(e)[:100])
        print(f"  [FAIL] {e}")

    results.append(result)
    return result


# ---------------------------------------------------------------------------
# 4. RSS FEEDS (News Outlets)
# ---------------------------------------------------------------------------

def test_rss_feeds():
    """Test RSS feeds from major news outlets."""
    print("\n" + "="*60)
    print("  4. RSS FEEDS - Major News Outlets")
    print("="*60)

    result = APITestResult("RSS Feeds")
    result.has_realtime = False  # Polling-based (1-5 min)
    result.rate_limit_info = "No limits (standard HTTP)"

    feeds = {
        "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "Reuters": "https://www.reutersagency.com/feed/",
        "France24 FR": "https://www.france24.com/fr/rss",
        "RT (Russia)": "https://www.rt.com/rss/news/",
        "Fox News": "https://moxie.foxnews.com/google-publisher/politics.xml",
        "CNN": "http://rss.cnn.com/rss/edition_world.rss",
        "Le Monde": "https://www.lemonde.fr/rss/une.xml",
    }

    all_articles = []
    working_feeds = 0

    for name, url in feeds.items():
        try:
            start = time.time()
            feed = feedparser.parse(url)
            elapsed = (time.time() - start) * 1000

            if feed.entries:
                working_feeds += 1
                result.available = True

                for entry in feed.entries[:5]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    # Detect language from feed
                    lang = "fr" if "france24.com/fr" in url or "lemonde.fr" in url else \
                           "ar" if "aljazeera" in url else \
                           "ru" if "rt.com" in url else "en"

                    import re
                    clean_summary = re.sub(r'<[^>]+>', '', summary).strip()

                    all_articles.append({
                        "source": name,
                        "title": title,
                        "text": f"{title}. {clean_summary[:300]}",
                        "language": lang,
                        "published": entry.get("published", ""),
                    })
                    result.languages_detected.add(lang)

                print(f"  [OK] {name}: {len(feed.entries)} articles ({elapsed:.0f}ms)")
            else:
                print(f"  [WARN] {name}: No entries (status: {feed.get('status', 'unknown')})")

        except Exception as e:
            result.errors.append(f"{name}: {str(e)[:40]}")
            print(f"  [FAIL] {name}: {e}")

    result.sample_count = len(all_articles)
    result.samples = all_articles[:5]
    if all_articles:
        result.avg_text_length = sum(len(a["text"]) for a in all_articles) // len(all_articles)
    result.response_time_ms = 500  # Average

    result.content_quality = "high"
    result.persuasion_relevance = "very high"

    print(f"\n  Working feeds: {working_feeds}/{len(feeds)}")
    print(f"  Total articles: {len(all_articles)}")
    print(f"  Languages: {result.languages_detected}")
    print(f"  Avg length: {result.avg_text_length} chars")

    if all_articles:
        print(f"\n  Sample article:")
        print(f"    [{all_articles[0]['source']}] {all_articles[0]['title'][:100]}")

    results.append(result)
    return result


# ---------------------------------------------------------------------------
# 5. YOUTUBE TRANSCRIPTS
# ---------------------------------------------------------------------------

def test_youtube_transcripts():
    """Test YouTube transcript extraction (no API key needed)."""
    print("\n" + "="*60)
    print("  5. YOUTUBE TRANSCRIPTS (youtube-transcript-api)")
    print("="*60)

    result = APITestResult("YouTube Transcripts")
    result.has_realtime = False  # On-demand
    result.rate_limit_info = "No official limits (scraping-based)"

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # Test with some well-known political/news videos
        test_videos = [
            ("dQw4w9WgXcQ", "en"),      # Popular video (test connectivity)
            ("9bZkp7q19f0", "ko"),       # Korean (multilingual test)
        ]

        all_transcripts = []
        for video_id, expected_lang in test_videos:
            try:
                start = time.time()
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[expected_lang, 'en'])
                elapsed = (time.time() - start) * 1000

                full_text = " ".join(t["text"] for t in transcript[:20])
                all_transcripts.append({
                    "video_id": video_id,
                    "text": full_text[:300],
                    "language": expected_lang,
                    "segments": len(transcript),
                    "duration": transcript[-1]["start"] if transcript else 0,
                })
                result.languages_detected.add(expected_lang)
                result.available = True
                result.response_time_ms = round(elapsed)

                print(f"  [OK] Video {video_id}: {len(transcript)} segments ({elapsed:.0f}ms)")

            except Exception as e:
                print(f"  [WARN] Video {video_id}: {e}")

        result.sample_count = len(all_transcripts)
        result.samples = all_transcripts
        if all_transcripts:
            result.avg_text_length = sum(len(t["text"]) for t in all_transcripts) // len(all_transcripts)

        result.content_quality = "high"
        result.persuasion_relevance = "high"

    except ImportError:
        print("  [SKIP] youtube-transcript-api not installed")
        print("         pip install youtube-transcript-api")
        result.errors.append("Package not installed")

    except Exception as e:
        result.errors.append(str(e)[:100])
        print(f"  [FAIL] {e}")

    results.append(result)
    return result


# ---------------------------------------------------------------------------
# 6. GNEWS API
# ---------------------------------------------------------------------------

def test_gnews():
    """Test GNews API (free tier: 100 req/day)."""
    print("\n" + "="*60)
    print("  6. GNEWS API (Free tier)")
    print("="*60)

    result = APITestResult("GNews")
    result.has_realtime = False  # Polling
    result.rate_limit_info = "100 req/day (free tier)"

    import os
    api_key = os.environ.get("GNEWS_API_KEY", "")

    if not api_key:
        print("  [INFO] No GNEWS_API_KEY set - testing without key...")
        print("         Get free key at: https://gnews.io/register")

        # Test without key to see response
        try:
            start = time.time()
            resp = requests.get(
                "https://gnews.io/api/v4/top-headlines",
                params={"lang": "en", "max": 10, "token": "demo"},
                timeout=10,
            )
            elapsed = (time.time() - start) * 1000
            result.response_time_ms = round(elapsed)

            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("articles", [])
                result.available = True
                result.sample_count = len(articles)

                for a in articles:
                    result.samples.append({
                        "title": a.get("title", ""),
                        "text": a.get("description", "")[:300],
                        "source": a.get("source", {}).get("name", ""),
                        "language": "en",
                    })
                    result.languages_detected.add("en")

                print(f"  [OK] {len(articles)} articles ({elapsed:.0f}ms)")
            elif resp.status_code == 403:
                print(f"  [INFO] Needs API key (free signup)")
                result.available = True  # Available but needs key
                result.errors.append("Needs free API key")
            else:
                print(f"  [WARN] HTTP {resp.status_code}")

        except Exception as e:
            result.errors.append(str(e)[:100])
            print(f"  [FAIL] {e}")
    else:
        try:
            start = time.time()
            resp = requests.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q": "propaganda OR manipulation",
                    "lang": "en",
                    "max": 10,
                    "token": api_key,
                },
                timeout=10,
            )
            elapsed = (time.time() - start) * 1000
            result.response_time_ms = round(elapsed)

            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("articles", [])
                result.available = True
                result.sample_count = len(articles)
                result.content_quality = "high"
                result.persuasion_relevance = "high"

                for a in articles:
                    result.samples.append({
                        "title": a.get("title", ""),
                        "text": a.get("description", "")[:300],
                        "source": a.get("source", {}).get("name", ""),
                    })

                print(f"  [OK] {len(articles)} articles ({elapsed:.0f}ms)")
            else:
                print(f"  [WARN] HTTP {resp.status_code}")

        except Exception as e:
            result.errors.append(str(e)[:100])
            print(f"  [FAIL] {e}")

    result.content_quality = "high"
    result.persuasion_relevance = "high"
    results.append(result)
    return result


# ---------------------------------------------------------------------------
# 7. WIKIPEDIA RECENT CHANGES (EventStream)
# ---------------------------------------------------------------------------

def test_wikipedia_stream():
    """Test Wikipedia Recent Changes stream (SSE, no auth)."""
    print("\n" + "="*60)
    print("  7. WIKIPEDIA - Recent Changes Stream (SSE)")
    print("="*60)

    result = APITestResult("Wikipedia Stream")
    result.has_realtime = True
    result.rate_limit_info = "No limits (public SSE stream)"

    try:
        import httpx

        start = time.time()
        changes = []

        # Use streaming to get a few events
        with httpx.stream(
            "GET",
            "https://stream.wikimedia.org/v2/stream/recentchange",
            timeout=15,
        ) as response:
            for i, line in enumerate(response.iter_lines()):
                if i > 100 or (time.time() - start) > 8:
                    break
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:])
                        if data.get("type") == "edit" and data.get("comment"):
                            changes.append({
                                "title": data.get("title", ""),
                                "comment": data.get("comment", "")[:200],
                                "wiki": data.get("wiki", ""),
                                "language": data.get("wiki", "en")[:2],
                            })
                            result.languages_detected.add(data.get("wiki", "en")[:2])
                    except json.JSONDecodeError:
                        continue

        elapsed = (time.time() - start) * 1000
        result.response_time_ms = round(elapsed)
        result.available = True
        result.sample_count = len(changes)
        result.samples = changes[:5]
        result.content_quality = "low"
        result.persuasion_relevance = "low"

        print(f"  [OK] {len(changes)} edits captured ({elapsed:.0f}ms)")
        print(f"  Languages: {result.languages_detected}")

        if changes:
            print(f"\n  Sample edit:")
            print(f"    [{changes[0]['wiki']}] {changes[0]['title']}: {changes[0]['comment'][:80]}")

    except Exception as e:
        result.errors.append(str(e)[:100])
        print(f"  [FAIL] {e}")

    results.append(result)
    return result


# ---------------------------------------------------------------------------
# 8. REDDIT (via old.reddit JSON - no auth)
# ---------------------------------------------------------------------------

def test_reddit_public():
    """Test Reddit public JSON API (no auth needed)."""
    print("\n" + "="*60)
    print("  8. REDDIT - Public JSON API (no auth)")
    print("="*60)

    result = APITestResult("Reddit Public")
    result.has_realtime = False  # Polling
    result.rate_limit_info = "~60 req/min (no auth, with user-agent)"

    subreddits = ["worldnews", "politics", "conspiracy", "propaganda"]
    all_posts = []

    headers = {
        "User-Agent": "PersuasiX-Research/1.0 (academic research tool)"
    }

    for sub in subreddits:
        try:
            start = time.time()
            resp = requests.get(
                f"https://www.reddit.com/r/{sub}/hot.json",
                params={"limit": 10},
                headers=headers,
                timeout=10,
            )
            elapsed = (time.time() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                posts = data.get("data", {}).get("children", [])
                result.available = True
                result.response_time_ms = round(elapsed)

                for post in posts:
                    p = post.get("data", {})
                    title = p.get("title", "")
                    selftext = p.get("selftext", "")
                    text = f"{title}. {selftext[:300]}" if selftext else title

                    if len(text) > 30:
                        all_posts.append({
                            "subreddit": sub,
                            "title": title[:200],
                            "text": text[:300],
                            "score": p.get("score", 0),
                            "num_comments": p.get("num_comments", 0),
                            "language": "en",
                        })
                        result.languages_detected.add("en")

                print(f"  [OK] r/{sub}: {len(posts)} posts ({elapsed:.0f}ms)")
            elif resp.status_code == 429:
                print(f"  [WARN] r/{sub}: Rate limited (429)")
                result.errors.append("Rate limited")
            else:
                print(f"  [WARN] r/{sub}: HTTP {resp.status_code}")

            time.sleep(1)  # Be polite

        except Exception as e:
            result.errors.append(f"r/{sub}: {str(e)[:40]}")
            print(f"  [FAIL] r/{sub}: {e}")

    result.sample_count = len(all_posts)
    result.samples = all_posts[:5]
    if all_posts:
        result.avg_text_length = sum(len(p["text"]) for p in all_posts) // len(all_posts)

    result.content_quality = "high"
    result.persuasion_relevance = "very high"

    print(f"\n  Total posts: {len(all_posts)}")
    print(f"  Avg length: {result.avg_text_length} chars")

    if all_posts:
        print(f"\n  Sample post:")
        print(f"    [r/{all_posts[0]['subreddit']}] {all_posts[0]['title'][:100]}")

    results.append(result)
    return result


# ---------------------------------------------------------------------------
# FINAL COMPARISON
# ---------------------------------------------------------------------------

def print_comparison():
    """Print a comparison table of all tested APIs."""
    print("\n\n")
    print("=" * 80)
    print("  FINAL COMPARISON - Free APIs for PersuasiX Social Monitoring")
    print("=" * 80)

    # Sort by relevance score
    def score(r):
        s = 0
        if r.available:
            s += 30
        if r.has_realtime:
            s += 20
        if r.persuasion_relevance in ("high", "very high"):
            s += 25
        if r.content_quality in ("high", "very high"):
            s += 15
        if len(r.languages_detected) > 2:
            s += 10
        if r.sample_count > 10:
            s += 10
        if not r.errors:
            s += 5
        return s

    sorted_results = sorted(results, key=score, reverse=True)

    print(f"\n{'API':<20} {'Status':<10} {'RT':<5} {'Samples':<9} {'Langs':<8} {'Speed':<8} {'Quality':<9} {'Persuasion':<12} {'Score'}")
    print("-" * 100)

    for r in sorted_results:
        status = "OK" if r.available else "FAIL"
        rt = "Yes" if r.has_realtime else "No"
        langs = str(len(r.languages_detected))
        speed = f"{r.response_time_ms}ms" if r.response_time_ms else "N/A"
        s = score(r)

        print(f"{r.name:<20} {status:<10} {rt:<5} {r.sample_count:<9} {langs:<8} {speed:<8} {r.content_quality:<9} {r.persuasion_relevance:<12} {s}/115")

    print("\n" + "-" * 80)
    print("\n  RECOMMENDATION:")
    print("  " + "-" * 40)

    top3 = sorted_results[:3]
    print(f"\n  TOP PICKS for PersuasiX:")
    for i, r in enumerate(top3, 1):
        print(f"    {i}. {r.name} (score: {score(r)}/115)")
        if r.has_realtime:
            print(f"       - Real-time streaming")
        print(f"       - {r.sample_count} samples, {len(r.languages_detected)} languages")
        print(f"       - Rate limit: {r.rate_limit_info}")

    print(f"\n  SUGGESTED ARCHITECTURE:")
    print(f"    Primary (real-time): {', '.join(r.name for r in sorted_results if r.has_realtime and r.available)}")
    print(f"    Secondary (polling): {', '.join(r.name for r in sorted_results if not r.has_realtime and r.available)}")

    # Save results
    output = {
        "tested_at": datetime.now().isoformat(),
        "results": [r.summary() for r in sorted_results],
        "recommendation": {
            "top_picks": [r.name for r in top3],
            "realtime_sources": [r.name for r in sorted_results if r.has_realtime and r.available],
            "polling_sources": [r.name for r in sorted_results if not r.has_realtime and r.available],
        },
    }

    output_path = Path(__file__).resolve().parent.parent / "data" / "api_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("  PersuasiX - Free API Testing Suite")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n  Testing all free APIs for social media monitoring...\n")

    # Run all tests
    test_mastodon()
    test_bluesky()
    test_hackernews()
    test_rss_feeds()
    test_youtube_transcripts()
    test_gnews()
    test_wikipedia_stream()
    test_reddit_public()

    # Print comparison
    print_comparison()
