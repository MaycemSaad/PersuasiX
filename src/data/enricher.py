"""Stage 3 — Enrich data with LLM-generated explanations, translations, and neutral rewrites."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()


ENRICHMENT_PROMPT = """\
You are an expert in rhetoric, propaganda analysis, and media literacy.

Given the following text and the persuasion techniques it uses, produce a JSON object with:
1. "explanation_en": A 2-3 sentence explanation in English of WHY each technique is manipulative in this context.
2. "explanation_fr": The same explanation in French.
3. "explanation_ar": The same explanation in Arabic.
4. "neutral_rewrite": A neutral, factual rewrite of the original text that removes ALL persuasion techniques while preserving the core factual claim. Same language as the original.
5. "severity": A score from 1 (mildly persuasive) to 5 (heavily manipulative).
6. "target_audience": Who is this manipulation targeting? (e.g., "general public", "voters", "consumers")

TEXT: {text}
LANGUAGE: {language}
TECHNIQUES DETECTED: {techniques}

Respond ONLY with valid JSON. No markdown, no extra text.
"""

NEUTRAL_TEXT_PROMPT = """\
You are an expert in rhetoric. Confirm that the following text is neutral and objective.
Return a JSON object with:
1. "is_neutral": true/false
2. "explanation_en": Brief explanation in English.
3. "explanation_fr": Brief explanation in French.
4. "explanation_ar": Brief explanation in Arabic.
5. "severity": 0

TEXT: {text}
LANGUAGE: {language}

Respond ONLY with valid JSON.
"""


class LLMEnricher:
    """Enrich dataset records with LLM-generated metadata."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
        max_retries: int = 3,
        temperature: float = 0.3,
    ) -> None:
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = self._init_client()

    def _init_client(self) -> Any:
        if self.provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI()
            except Exception as e:
                logger.warning(f"OpenAI client init failed: {e}. Using offline mode.")
                return None
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                return Anthropic()
            except Exception as e:
                logger.warning(f"Anthropic client init failed: {e}. Using offline mode.")
                return None
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich every row with explanations, rewrites, and severity scores."""
        logger.info(f"Enriching {len(df)} rows with {self.provider}/{self.model} …")

        enriched_rows: list[dict] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Enriching"):
            enriched = self._enrich_single(row)
            enriched_rows.append(enriched)

        result = pd.DataFrame(enriched_rows)
        logger.success(f"Enrichment complete: {len(result)} rows")
        return result

    # ------------------------------------------------------------------
    # Single-row enrichment
    # ------------------------------------------------------------------

    def _enrich_single(self, row: pd.Series) -> dict:
        base = row.to_dict()
        techniques = base.get("techniques", [])
        is_persuasive = base.get("is_persuasive", bool(techniques))

        if is_persuasive and techniques:
            prompt = ENRICHMENT_PROMPT.format(
                text=base["text"],
                language=base.get("language", "en"),
                techniques=", ".join(techniques),
            )
        else:
            prompt = NEUTRAL_TEXT_PROMPT.format(
                text=base["text"],
                language=base.get("language", "en"),
            )

        llm_output = self._call_llm(prompt)
        if llm_output:
            base.update(llm_output)
        else:
            base.update(self._fallback_enrichment(base))

        return base

    def _call_llm(self, prompt: str) -> dict | None:
        if self._client is None:
            return None

        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=1024,
                    )
                    content = response.choices[0].message.content.strip()
                elif self.provider == "anthropic":
                    response = self._client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = response.content[0].text.strip()
                else:
                    return None

                content = content.removeprefix("```json").removesuffix("```").strip()
                return json.loads(content)

            except json.JSONDecodeError:
                logger.warning(f"JSON parse error on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

        return None

    # ------------------------------------------------------------------
    # Fallback when LLM is unavailable
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_enrichment(row: dict) -> dict:
        techniques = row.get("techniques", [])
        is_persuasive = row.get("is_persuasive", False)

        if is_persuasive and techniques:
            tech_str = ", ".join(t.replace("_", " ") for t in techniques)
            return {
                "explanation_en": f"This text uses the following persuasion techniques: {tech_str}.",
                "explanation_fr": f"Ce texte utilise les techniques de persuasion suivantes : {tech_str}.",
                "explanation_ar": f"يستخدم هذا النص تقنيات الإقناع التالية: {tech_str}.",
                "neutral_rewrite": "[Neutral rewrite requires LLM enrichment]",
                "severity": min(len(techniques), 5),
                "target_audience": "general public",
            }
        return {
            "explanation_en": "This text appears to be neutral and objective.",
            "explanation_fr": "Ce texte semble neutre et objectif.",
            "explanation_ar": "يبدو هذا النص محايداً وموضوعياً.",
            "is_neutral": True,
            "severity": 0,
        }
