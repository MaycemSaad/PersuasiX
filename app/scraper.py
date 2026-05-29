"""URL scraping and article extraction for PersuasiX."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger


@dataclass
class ScrapedArticle:
    """Container for a scraped article."""
    url: str
    title: str
    text: str
    author: str
    source: str
    date: str
    language: str
    word_count: int
    success: bool
    error: str = ""

    def short_title(self, max_len: int = 60) -> str:
        if len(self.title) <= max_len:
            return self.title
        return self.title[: max_len - 3] + "..."


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,ar;q=0.7",
}


def _detect_language(text: str) -> str:
    """Simple heuristic language detection."""
    arabic_ratio = len(re.findall(r"[؀-ۿ]", text)) / max(len(text), 1)
    if arabic_ratio > 0.3:
        return "ar"
    french_keywords = {"le", "la", "les", "de", "du", "des", "un", "une", "et", "est", "dans", "pour", "qui", "que", "ce", "cette", "sur", "avec", "pas", "sont"}
    words = set(re.findall(r"\b\w+\b", text.lower()))
    french_hits = len(words & french_keywords)
    if french_hits >= 4:
        return "fr"
    return "en"


def _extract_domain(url: str) -> str:
    """Extract domain name from URL."""
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else url


def scrape_url(url: str) -> ScrapedArticle:
    """Scrape an article from a URL. Uses trafilatura with requests+BS4 fallback."""
    if not url or not url.strip():
        return ScrapedArticle(
            url=url, title="", text="", author="", source="", date="",
            language="en", word_count=0, success=False, error="No URL provided.",
        )

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    source = _extract_domain(url)

    # --- Strategy 1: trafilatura ---
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            result = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                output_format="txt",
            )
            metadata = trafilatura.extract_metadata(downloaded)

            if result and len(result) > 100:
                title = metadata.title if metadata and metadata.title else ""
                author = metadata.author if metadata and metadata.author else ""
                date = ""
                if metadata and metadata.date:
                    date = str(metadata.date)

                lang = _detect_language(result)
                words = len(result.split())
                logger.info(f"Scraped {source}: {words} words via trafilatura")

                return ScrapedArticle(
                    url=url, title=title, text=result, author=author,
                    source=source, date=date, language=lang,
                    word_count=words, success=True,
                )
    except ImportError:
        logger.warning("trafilatura not installed, trying fallback")
    except Exception as e:
        logger.warning(f"trafilatura failed: {e}")

    # --- Strategy 2: requests + BeautifulSoup ---
    try:
        from bs4 import BeautifulSoup

        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)

        if len(text) > 100:
            lang = _detect_language(text)
            words = len(text.split())
            logger.info(f"Scraped {source}: {words} words via BS4 fallback")
            return ScrapedArticle(
                url=url, title=title, text=text, author="",
                source=source, date="", language=lang,
                word_count=words, success=True,
            )
    except ImportError:
        logger.warning("beautifulsoup4 not installed")
    except Exception as e:
        logger.warning(f"BS4 fallback failed: {e}")

    # --- Strategy 3: raw requests ---
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        raw = re.sub(r"<[^>]+>", " ", resp.text)
        raw = re.sub(r"\s+", " ", raw).strip()
        if len(raw) > 200:
            text = raw[:5000]
            lang = _detect_language(text)
            return ScrapedArticle(
                url=url, title=source, text=text, author="",
                source=source, date="", language=lang,
                word_count=len(text.split()), success=True,
            )
    except Exception as e:
        logger.error(f"All scraping strategies failed for {url}: {e}")

    return ScrapedArticle(
        url=url, title="", text="", author="", source=source, date="",
        language="en", word_count=0, success=False,
        error=f"Could not extract content from {source}. The site may block automated access.",
    )
