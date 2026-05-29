"""Stage 2 — Clean and normalize collected texts."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
from loguru import logger


class DataCleaner:
    """Deduplicate, normalize, and filter raw data."""

    def __init__(self, min_length: int = 20, max_length: int = 2048) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self._url_re = re.compile(r"https?://\S+")
        self._multi_space_re = re.compile(r"[ \t]+")
        self._multi_newline_re = re.compile(r"\n{3,}")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline."""
        initial = len(df)
        logger.info(f"Cleaning {initial} rows …")

        df = df.copy()
        df["text"] = df["text"].apply(self._normalize_text)
        df = self._drop_duplicates(df)
        df = self._filter_length(df)
        df = self._validate_language(df)
        df = df.reset_index(drop=True)

        logger.success(f"Cleaning done: {initial} → {len(df)} rows ({initial - len(df)} removed)")
        return df

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = self._url_re.sub("[URL]", text)
        text = self._multi_space_re.sub(" ", text)
        text = self._multi_newline_re.sub("\n\n", text)
        text = text.strip()
        return text

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["_hash"] = df["text"].apply(lambda t: hashlib.md5(t.encode()).hexdigest())
        before = len(df)
        df = df.drop_duplicates(subset="_hash").drop(columns="_hash")
        removed = before - len(df)
        if removed:
            logger.info(f"Removed {removed} exact duplicates")
        return df

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_length(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["text"].str.len().between(self.min_length, self.max_length)
        removed = (~mask).sum()
        if removed:
            logger.info(f"Removed {removed} rows outside length bounds [{self.min_length}, {self.max_length}]")
        return df[mask]

    def _validate_language(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_langs = {"en", "fr", "ar"}
        if "language" in df.columns:
            mask = df["language"].isin(valid_langs)
            removed = (~mask).sum()
            if removed:
                logger.info(f"Removed {removed} rows with unsupported language")
            return df[mask]
        return df
