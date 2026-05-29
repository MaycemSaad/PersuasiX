"""Stage 1 — Collect raw texts from multiple sources for persuasion analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm


SEMEVAL_PROPAGANDA_URL = (
    "https://propaganda.qcri.org/semeval2020-task11/"
)

TECHNIQUE_LABELS: list[str] = [
    "appeal_to_fear",
    "appeal_to_authority",
    "bandwagon",
    "false_dilemma",
    "ad_hominem",
    "straw_man",
    "red_herring",
    "loaded_language",
    "whataboutism",
    "causal_oversimplification",
    "appeal_to_emotion",
    "repetition",
    "exaggeration",
    "doubt",
    "slogans",
    "name_calling",
    "flag_waving",
    "thought_terminating_cliche",
]


class DataCollector:
    """Collect persuasive and neutral texts from heterogeneous sources."""

    def __init__(self, output_dir: str | Path, rate_limit: float = 1.0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_all(self) -> pd.DataFrame:
        """Run every collector and merge into a single DataFrame."""
        frames: list[pd.DataFrame] = []

        for name, gen in [
            ("semeval_propaganda", self._collect_semeval_samples()),
            ("synthetic_persuasive", self._generate_synthetic_persuasive()),
            ("synthetic_neutral", self._generate_synthetic_neutral()),
        ]:
            rows = list(gen)
            if rows:
                df = pd.DataFrame(rows)
                df["source"] = name
                frames.append(df)
                logger.info(f"Collected {len(df)} rows from {name}")

        if not frames:
            logger.warning("No data collected from any source")
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        out_path = self.output_dir / "raw_collected.json"
        merged.to_json(out_path, orient="records", lines=True, force_ascii=False)
        logger.success(f"Saved {len(merged)} total rows → {out_path}")
        return merged

    # ------------------------------------------------------------------
    # Source-specific collectors
    # ------------------------------------------------------------------

    def _collect_semeval_samples(self) -> Iterator[dict]:
        """Yield sample entries modelled on SemEval-2020 Task 11 propaganda corpus."""
        samples = [
            {
                "text": "If we don't act NOW, our children will inherit a wasteland. "
                        "Every scientist agrees — this is our last chance.",
                "language": "en",
                "techniques": ["appeal_to_fear", "appeal_to_authority", "exaggeration"],
                "is_persuasive": True,
            },
            {
                "text": "Studies show a 1.2°C average temperature increase over the past century, "
                        "with projections varying by model assumptions.",
                "language": "en",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "Everybody knows that this policy is the only way forward. "
                        "Are you really going to side with the enemies of progress?",
                "language": "en",
                "techniques": ["bandwagon", "false_dilemma", "loaded_language"],
                "is_persuasive": True,
            },
            {
                "text": "The proposed policy has both supporters and critics among economists.",
                "language": "en",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "Ne soyez pas naïfs ! Ce politicien est un menteur notoire qui ne cherche "
                        "qu'à remplir ses poches. Réveillez-vous !",
                "language": "fr",
                "techniques": ["name_calling", "ad_hominem", "loaded_language"],
                "is_persuasive": True,
            },
            {
                "text": "Le candidat a présenté son programme économique lors d'une conférence de presse.",
                "language": "fr",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "هذا القرار سيدمر مستقبل أطفالنا! كل الخبراء يؤكدون ذلك. "
                        "من يعارض هذا الرأي هو عدو للشعب.",
                "language": "ar",
                "techniques": ["appeal_to_fear", "appeal_to_authority", "false_dilemma"],
                "is_persuasive": True,
            },
            {
                "text": "صدر تقرير جديد عن المنظمة يتناول تأثير السياسات الاقتصادية على معدلات البطالة.",
                "language": "ar",
                "techniques": [],
                "is_persuasive": False,
            },
        ]
        yield from samples

    def _generate_synthetic_persuasive(self) -> Iterator[dict]:
        """Generate additional persuasive examples covering all techniques."""
        examples = [
            {
                "text": "Only a fool would oppose this bill. The silent majority stands behind it, "
                        "and history will judge those who stood in the way.",
                "language": "en",
                "techniques": ["name_calling", "bandwagon", "appeal_to_fear"],
                "is_persuasive": True,
            },
            {
                "text": "Why are we even talking about healthcare when the REAL issue is immigration? "
                        "That's what they don't want you to focus on.",
                "language": "en",
                "techniques": ["red_herring", "whataboutism", "doubt"],
                "is_persuasive": True,
            },
            {
                "text": "Our nation was built by heroes, not cowards. Support this measure "
                        "or betray everything our founders fought for!",
                "language": "en",
                "techniques": ["flag_waving", "false_dilemma", "loaded_language"],
                "is_persuasive": True,
            },
            {
                "text": "It's simple: more regulation means fewer jobs. Period. End of story.",
                "language": "en",
                "techniques": ["causal_oversimplification", "thought_terminating_cliche", "slogans"],
                "is_persuasive": True,
            },
            {
                "text": "Professor Smith from Harvard — a leading expert — confirms that "
                        "this product is safe. I repeat: it is SAFE.",
                "language": "en",
                "techniques": ["appeal_to_authority", "repetition", "exaggeration"],
                "is_persuasive": True,
            },
            {
                "text": "Think about the children crying at night because of this policy. "
                        "How can anyone with a heart support such cruelty?",
                "language": "en",
                "techniques": ["appeal_to_emotion", "loaded_language", "straw_man"],
                "is_persuasive": True,
            },
            {
                "text": "Si vous n'êtes pas avec nous, vous êtes contre nous. "
                        "Les vrais patriotes savent ce qu'il faut faire.",
                "language": "fr",
                "techniques": ["false_dilemma", "flag_waving", "bandwagon"],
                "is_persuasive": True,
            },
            {
                "text": "On nous cache la vérité ! Les élites manipulent tout. "
                        "Il est temps de se réveiller avant qu'il ne soit trop tard !",
                "language": "fr",
                "techniques": ["doubt", "appeal_to_fear", "loaded_language"],
                "is_persuasive": True,
            },
            {
                "text": "الجميع يعرف أن هذا هو الحل الوحيد. من يقول غير ذلك فهو جاهل لا يفهم شيئاً.",
                "language": "ar",
                "techniques": ["bandwagon", "name_calling", "false_dilemma"],
                "is_persuasive": True,
            },
            {
                "text": "علماؤنا الأفاضل أكدوا مراراً أن هذا المنتج آمن تماماً. "
                        "لا تصدقوا الأكاذيب التي ينشرها أعداء التقدم!",
                "language": "ar",
                "techniques": ["appeal_to_authority", "repetition", "name_calling"],
                "is_persuasive": True,
            },
        ]
        yield from examples

    def _generate_synthetic_neutral(self) -> Iterator[dict]:
        """Generate neutral/objective examples as negative samples."""
        examples = [
            {
                "text": "The committee voted 7-5 in favor of the amendment after three hours of debate.",
                "language": "en",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "According to the quarterly report, revenue increased by 3.2% compared to last year.",
                "language": "en",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "The new policy will take effect on January 1st, affecting approximately 2 million residents.",
                "language": "en",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "Researchers published their findings in the journal Nature, noting both strengths and limitations.",
                "language": "en",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "Le parlement a adopté la loi avec 312 voix pour et 245 contre.",
                "language": "fr",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "L'étude menée sur 5000 participants montre des résultats mitigés selon les tranches d'âge.",
                "language": "fr",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "أعلنت الحكومة عن خطة اقتصادية جديدة تتضمن عدة إصلاحات في قطاع التعليم.",
                "language": "ar",
                "techniques": [],
                "is_persuasive": False,
            },
            {
                "text": "بلغ عدد المشاركين في الاستطلاع خمسة آلاف شخص من مختلف المناطق.",
                "language": "ar",
                "techniques": [],
                "is_persuasive": False,
            },
        ]
        yield from examples

    # ------------------------------------------------------------------
    # Web scraping helpers
    # ------------------------------------------------------------------

    def collect_from_url(self, url: str, language: str = "en") -> dict | None:
        """Scrape a single article and return a raw record."""
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return None
            text = trafilatura.extract(downloaded)
            if text and len(text) >= 20:
                time.sleep(self.rate_limit)
                return {"text": text, "language": language, "url": url}
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
        return None

    def collect_from_urls(self, urls: list[str], language: str = "en") -> list[dict]:
        """Scrape multiple URLs."""
        results = []
        for url in tqdm(urls, desc="Scraping"):
            record = self.collect_from_url(url, language)
            if record:
                results.append(record)
        return results
