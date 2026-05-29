"""Tests for the data pipeline."""

import pandas as pd
import pytest

from src.data.collector import DataCollector, TECHNIQUE_LABELS
from src.data.cleaner import DataCleaner
from src.data.enricher import LLMEnricher
from src.data.validator import DataValidator


class TestDataCollector:
    def test_technique_labels_not_empty(self):
        assert len(TECHNIQUE_LABELS) == 18

    def test_technique_labels_unique(self):
        assert len(set(TECHNIQUE_LABELS)) == len(TECHNIQUE_LABELS)

    def test_collect_all_returns_dataframe(self, tmp_path):
        collector = DataCollector(output_dir=tmp_path / "raw")
        df = collector.collect_all()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_collected_data_has_required_columns(self, tmp_path):
        collector = DataCollector(output_dir=tmp_path / "raw")
        df = collector.collect_all()
        for col in ["text", "language", "techniques", "is_persuasive"]:
            assert col in df.columns

    def test_languages_covered(self, tmp_path):
        collector = DataCollector(output_dir=tmp_path / "raw")
        df = collector.collect_all()
        langs = set(df["language"])
        assert "en" in langs
        assert "fr" in langs
        assert "ar" in langs


class TestDataCleaner:
    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "text": ["hello world", "hello world", "unique text"],
            "language": ["en", "en", "en"],
        })
        cleaner = DataCleaner(min_length=5)
        result = cleaner.clean(df)
        assert len(result) == 2

    def test_filters_short_texts(self):
        df = pd.DataFrame({
            "text": ["hi", "this is a longer text that should pass the filter"],
            "language": ["en", "en"],
        })
        cleaner = DataCleaner(min_length=10)
        result = cleaner.clean(df)
        assert len(result) == 1

    def test_normalizes_whitespace(self):
        cleaner = DataCleaner()
        text = "  hello    world  \n\n\n\n  test  "
        normalized = cleaner._normalize_text(text)
        assert "    " not in normalized
        assert normalized.startswith("hello")


class TestLLMEnricher:
    def test_fallback_enrichment(self):
        row = {
            "text": "Test text",
            "techniques": ["appeal_to_fear", "loaded_language"],
            "is_persuasive": True,
            "language": "en",
        }
        result = LLMEnricher._fallback_enrichment(row)
        assert "explanation_en" in result
        assert "explanation_fr" in result
        assert "explanation_ar" in result
        assert result["severity"] == 2

    def test_fallback_neutral(self):
        row = {"text": "Neutral.", "techniques": [], "is_persuasive": False}
        result = LLMEnricher._fallback_enrichment(row)
        assert result["severity"] == 0
        assert result["is_neutral"] is True


class TestDataValidator:
    def test_validates_required_fields(self):
        df = pd.DataFrame({
            "text": ["valid text here", "", "another valid text"],
            "language": ["en", "en", "en"],
            "techniques": [["appeal_to_fear"], [], []],
            "is_persuasive": [True, False, False],
        })
        validator = DataValidator(use_embeddings=False)
        result = validator.validate(df)
        assert len(result) == 2

    def test_validates_technique_labels(self):
        df = pd.DataFrame({
            "text": ["text one", "text two"],
            "language": ["en", "en"],
            "techniques": [["appeal_to_fear"], ["INVALID_TECHNIQUE"]],
            "is_persuasive": [True, True],
            "explanation_en": ["explanation", "explanation"],
        })
        validator = DataValidator(use_embeddings=False)
        result = validator.validate(df)
        assert len(result) == 1
