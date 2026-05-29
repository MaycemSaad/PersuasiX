"""Tests for the end-to-end pipeline."""

import pytest
from src.pipeline.persuasix_pipeline import AnalysisResult


class TestAnalysisResult:
    def test_to_dict(self):
        result = AnalysisResult(
            text="test",
            language="en",
            is_persuasive=True,
            techniques=["appeal_to_fear"],
            technique_probabilities={"appeal_to_fear": 0.95},
            explanation="This text uses fear.",
            neutral_rewrite="A neutral version.",
            severity_score=3.5,
            manipulation_score=0.7,
            cross_lingual_explanations={"en": "Fear.", "fr": "Peur."},
        )
        d = result.to_dict()
        assert d["is_persuasive"] is True
        assert len(d["techniques"]) == 1
        assert d["severity_score"] == 3.5

    def test_neutral_result(self):
        result = AnalysisResult(
            text="The committee voted 7-5.",
            language="en",
            is_persuasive=False,
            techniques=[],
            technique_probabilities={},
            explanation="",
            neutral_rewrite="The committee voted 7-5.",
            severity_score=0.0,
            manipulation_score=0.0,
        )
        assert not result.is_persuasive
        assert result.severity_score == 0.0


class TestPersuasixPipeline:
    def test_import(self):
        from src.pipeline.persuasix_pipeline import PersuasixPipeline
        assert PersuasixPipeline is not None

    def test_pipeline_without_models(self):
        from src.pipeline.persuasix_pipeline import PersuasixPipeline
        pipe = PersuasixPipeline()
        result = pipe.analyze("test text")
        assert isinstance(result, AnalysisResult)
        assert not result.is_persuasive
