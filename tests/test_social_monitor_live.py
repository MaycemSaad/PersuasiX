"""
Live integration test for the updated SocialMonitor with free APIs.
Tests: Mastodon, RSS Feeds, YouTube Transcripts, HackerNews.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.social_monitor import (
    SocialMonitor,
    MastodonConnector,
    RSSConnector,
    YouTubeTranscriptConnector,
    HackerNewsConnector,
)


def test_mastodon():
    print("\n" + "=" * 60)
    print("  TEST: Mastodon Connector")
    print("=" * 60)

    conn = MastodonConnector()
    assert conn.is_available, "Mastodon should always be available"

    # 1. Public timeline
    print("\n  1. Public timeline...")
    posts = conn.get_public_timeline(limit=10)
    print(f"     -> {len(posts)} posts")
    if posts:
        p = posts[0]
        print(f"     Sample: [{p.language}] {p.text[:100]}...")
        assert p.platform == "mastodon"
        assert len(p.text) > 0

    # 2. Hashtag search
    print("\n  2. Hashtag search (#news)...")
    results = conn.search("news", limit=10)
    print(f"     -> {len(results)} posts")

    # 3. Trending
    print("\n  3. Trending posts...")
    trending = conn.get_trending(limit=5)
    print(f"     -> {len(trending)} trending")

    print("\n  [PASS] Mastodon connector works!")
    return True


def test_rss():
    print("\n" + "=" * 60)
    print("  TEST: RSS Connector")
    print("=" * 60)

    conn = RSSConnector()
    assert conn.is_available, "RSS should always be available"

    # 1. Fetch all
    print("\n  1. Fetch all feeds...")
    posts = conn.fetch_all(limit_per_feed=3)
    print(f"     -> {len(posts)} articles")
    langs = set(p.language for p in posts)
    print(f"     Languages: {langs}")

    if posts:
        p = posts[0]
        print(f"     Sample: [{p.metadata.get('feed_name')}] {p.text[:100]}...")
        assert p.platform == "rss"

    # 2. Fetch by language
    print("\n  2. French articles only...")
    fr_posts = conn.fetch_by_language("fr", limit=5)
    print(f"     -> {len(fr_posts)} French articles")

    # 3. Search
    print("\n  3. Search for 'war OR conflict'...")
    matched = conn.search_in_feeds("war OR conflict")
    print(f"     -> {len(matched)} matched articles")

    # 4. Feed list
    feeds = conn.get_feed_list()
    print(f"\n  4. {len(feeds)} feeds configured")

    # 5. Custom feed
    print("\n  5. Add custom feed...")
    conn.add_feed("TechCrunch", "https://techcrunch.com/feed/", "en")
    assert "TechCrunch" in conn.feeds

    print("\n  [PASS] RSS connector works!")
    return True


def test_youtube_transcripts():
    print("\n" + "=" * 60)
    print("  TEST: YouTube Transcript Connector")
    print("=" * 60)

    conn = YouTubeTranscriptConnector()

    if not conn.is_available:
        print("  [SKIP] youtube-transcript-api not installed")
        return False

    # 1. Get transcript
    print("\n  1. Fetching transcript (jNQXAC9IVRw)...")
    posts = conn.get_transcript("jNQXAC9IVRw", languages=["en"])
    print(f"     -> {len(posts)} chunks")

    if posts:
        p = posts[0]
        print(f"     Sample: {p.text[:100]}...")
        print(f"     Time: {p.timestamp}")
        assert p.platform == "youtube_transcript"

    # 2. Available languages
    print("\n  2. Available languages...")
    langs = conn.get_available_languages("jNQXAC9IVRw")
    print(f"     -> {langs}")

    print("\n  [PASS] YouTube Transcript connector works!")
    return True


def test_hackernews():
    print("\n" + "=" * 60)
    print("  TEST: HackerNews Connector")
    print("=" * 60)

    conn = HackerNewsConnector()
    assert conn.is_available

    # 1. Top stories
    print("\n  1. Top stories + comments...")
    posts = conn.get_top_stories(limit=5, include_comments=True)
    stories = [p for p in posts if p.metadata.get("type") == "story"]
    comments = [p for p in posts if p.metadata.get("type") == "comment"]
    print(f"     -> {len(stories)} stories, {len(comments)} comments")

    if stories:
        s = stories[0]
        print(f"     Top story: {s.text[:100]}...")
        print(f"     Score: {s.engagement.get('score', 0)}")
        assert s.platform == "hackernews"

    # 2. New stories
    print("\n  2. New stories...")
    new = conn.get_new_stories(limit=5)
    print(f"     -> {len(new)} new stories")

    print("\n  [PASS] HackerNews connector works!")
    return True


def test_unified_monitor():
    print("\n" + "=" * 60)
    print("  TEST: Unified SocialMonitor")
    print("=" * 60)

    monitor = SocialMonitor(pipeline=None)  # No pipeline, just test fetching

    # 1. Platform status
    print("\n  1. Platform status:")
    status = monitor.get_platform_status()
    for name, info in status.items():
        icon = "[FREE]" if info["type"] == "free" else "[PAID]"
        avail = "OK" if info["available"] else "N/A"
        rt = "RT" if info.get("realtime") else "  "
        print(f"     {icon} {name:<22} {avail:<5} {rt}")

    # 2. Multi-platform search (fetch only, no analysis)
    print("\n  2. Multi-platform search (no analysis)...")
    results = monitor.multi_platform_search(
        "politics",
        platforms=["mastodon", "rss", "hackernews"],
        analyze=False,
    )

    total = 0
    for platform, posts in results.items():
        count = len(posts)
        total += count
        print(f"     {platform}: {count} posts")

    print(f"     TOTAL: {total} posts across {len(results)} platforms")

    print("\n  [PASS] Unified monitor works!")
    return True


def test_report_generation():
    print("\n" + "=" * 60)
    print("  TEST: Report Generation (mock data)")
    print("=" * 60)

    from src.pipeline.social_monitor import SocialPost, SocialAnalysisResult

    # Create mock results
    mock_results = []
    for i in range(10):
        post = SocialPost(
            platform=["mastodon", "rss", "hackernews"][i % 3],
            post_id=f"test_{i}",
            text=f"Test post {i}",
            language=["en", "fr", "ar"][i % 3],
        )
        result = SocialAnalysisResult(
            post=post,
            is_persuasive=i % 3 == 0,
            manipulation_score=0.3 + (i * 0.07),
            techniques=["Loaded Language", "Appeal to Fear"][:i % 3 + 1] if i % 3 == 0 else [],
        )
        mock_results.append(result)

    monitor = SocialMonitor(pipeline=None)
    report = monitor.generate_report(mock_results)

    print(f"  Total: {report['total_posts']}")
    print(f"  Manipulative: {report['manipulative_count']} ({report['manipulation_rate']}%)")
    print(f"  Avg score: {report['avg_manipulation_score']}")
    print(f"  Platform breakdown: {report['platform_breakdown']}")
    print(f"  Language breakdown: {report['language_breakdown']}")
    print(f"  Techniques: {report['technique_frequency']}")

    assert report["total_posts"] == 10
    print("\n  [PASS] Report generation works!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  PersuasiX Social Monitor - Live Integration Test")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    tests = [
        ("Mastodon", test_mastodon),
        ("RSS Feeds", test_rss),
        ("YouTube Transcripts", test_youtube_transcripts),
        ("HackerNews", test_hackernews),
        ("Unified Monitor", test_unified_monitor),
        ("Report Generation", test_report_generation),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
