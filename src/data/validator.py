"""Stage 4 — Validate enriched data with semantic similarity checks and rule-based filters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class DataValidator:
    """Quality-gate: semantic checks, completeness, and consistency."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        use_embeddings: bool = True,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self.use_embeddings = False
        return self._embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all validation checks and return the cleaned DataFrame."""
        initial = len(df)
        logger.info(f"Validating {initial} rows …")

        df = df.copy()
        df = self._check_required_fields(df)
        df = self._check_technique_labels(df)
        df = self._check_explanation_quality(df)

        if self.use_embeddings and "neutral_rewrite" in df.columns:
            df = self._check_semantic_similarity(df)

        df["quality_score"] = self._compute_quality_scores(df)
        df = df.reset_index(drop=True)

        logger.success(f"Validation done: {initial} → {len(df)} rows")
        self._log_stats(df)
        return df

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_required_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["text", "language", "techniques", "is_persuasive"]
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                df[col] = None

        mask = df["text"].notna() & (df["text"].str.len() > 0)
        removed = (~mask).sum()
        if removed:
            logger.info(f"Removed {removed} rows with empty text")
        return df[mask]

    def _check_technique_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        from .collector import TECHNIQUE_LABELS

        valid = set(TECHNIQUE_LABELS)

        def validate_techniques(row):
            techs = row.get("techniques", [])
            if not isinstance(techs, list):
                return False
            if row.get("is_persuasive") and not techs:
                return False
            return all(t in valid for t in techs)

        mask = df.apply(validate_techniques, axis=1)
        removed = (~mask).sum()
        if removed:
            logger.info(f"Removed {removed} rows with invalid technique labels")
        return df[mask]

    def _check_explanation_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        if "explanation_en" not in df.columns:
            return df

        persuasive = df["is_persuasive"] == True  # noqa: E712
        has_explanation = df["explanation_en"].notna() & (df["explanation_en"].str.len() > 10)

        mask = ~persuasive | has_explanation
        removed = (~mask).sum()
        if removed:
            logger.info(f"Removed {removed} persuasive rows without explanations")
        return df[mask]

    def _check_semantic_similarity(self, df: pd.DataFrame) -> pd.DataFrame:
        embedder = self._get_embedder()
        if embedder is None:
            return df

        has_rewrite = (
            df["neutral_rewrite"].notna()
            & (df["neutral_rewrite"] != "[Neutral rewrite requires LLM enrichment]")
        )
        subset = df[has_rewrite].copy()

        if subset.empty:
            return df

        logger.info(f"Computing semantic similarity for {len(subset)} rewrite pairs …")
        originals = subset["text"].tolist()
        rewrites = subset["neutral_rewrite"].tolist()

        emb_orig = embedder.encode(originals, show_progress_bar=False)
        emb_rew = embedder.encode(rewrites, show_progress_bar=False)

        cos_sim = np.array([
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            for a, b in zip(emb_orig, emb_rew)
        ])

        df.loc[has_rewrite, "rewrite_similarity"] = cos_sim

        low_sim = cos_sim < self.similarity_threshold
        if low_sim.any():
            logger.warning(
                f"{low_sim.sum()} rewrites below similarity threshold "
                f"({self.similarity_threshold}); flagging for review"
            )
            df.loc[subset.index[low_sim], "needs_review"] = True

        return df

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def _compute_quality_scores(self, df: pd.DataFrame) -> pd.Series:
        scores = pd.Series(1.0, index=df.index)

        has_explanation = df.get("explanation_en", pd.Series()).notna()
        scores += has_explanation.astype(float) * 0.3

        has_rewrite = (
            df.get("neutral_rewrite", pd.Series()).notna()
            & (df.get("neutral_rewrite", pd.Series()) != "[Neutral rewrite requires LLM enrichment]")
        )
        scores += has_rewrite.astype(float) * 0.3

        if "rewrite_similarity" in df.columns:
            sim = df["rewrite_similarity"].fillna(0)
            scores += (sim >= self.similarity_threshold).astype(float) * 0.2

        has_multilingual = (
            df.get("explanation_fr", pd.Series()).notna()
            & df.get("explanation_ar", pd.Series()).notna()
        )
        scores += has_multilingual.astype(float) * 0.2

        return scores.clip(0, 2)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _log_stats(df: pd.DataFrame) -> None:
        total = len(df)
        persuasive = df["is_persuasive"].sum()
        neutral = total - persuasive
        langs = df["language"].value_counts().to_dict()
        avg_quality = df["quality_score"].mean()

        logger.info(
            f"Dataset stats: {total} total | {persuasive} persuasive | "
            f"{neutral} neutral | languages: {langs} | avg quality: {avg_quality:.2f}"
        )
