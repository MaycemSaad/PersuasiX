"""End-to-end PersuasiX pipeline: Text → Detection → Classification → Explanation → Neutralization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from ..data.collector import TECHNIQUE_LABELS
from ..models.detector import PersuasionDetector
from ..models.explainer import PersuasionExplainer
from ..models.neutralizer import TextNeutralizer
from ..models.scorer import ReliabilityScorer

try:
    from .span_detector import HybridSpanDetector
    HAS_SPAN_DETECTOR = True
except ImportError:
    HAS_SPAN_DETECTOR = False

try:
    from .fact_checker import PersuasixFactChecker
    HAS_FACT_CHECKER = True
except ImportError:
    HAS_FACT_CHECKER = False


ANALYSIS_PROMPT = """\
You are PersuasiX, an expert NLP system specialized in detecting, classifying, \
explaining, and neutralizing persuasion and manipulation techniques in text.

## Task
Analyze the input text for persuasion/manipulation techniques. \
Be precise and evidence-based. Do NOT over-detect — only flag techniques you are confident about.

## Technique Catalog (ONLY use techniques from this list)
1. appeal_to_fear — Instills fear to influence the audience
2. appeal_to_authority — Uses authority figures without proper evidence
3. bandwagon — Implies everyone agrees, pressuring conformity
4. false_dilemma — Presents only two options when more exist
5. ad_hominem — Attacks the person rather than the argument
6. straw_man — Misrepresents someone's argument to attack it
7. red_herring — Introduces irrelevant topics to divert attention
8. loaded_language — Uses emotionally charged words to manipulate
9. whataboutism — Deflects criticism by pointing to others' faults
10. causal_oversimplification — Reduces complex issues to a single cause
11. appeal_to_emotion — Exploits emotions instead of logic
12. repetition — Repeats a message to make it seem more true
13. exaggeration — Overstates facts for stronger impression
14. doubt — Questions credibility without evidence
15. slogans — Uses catchy phrases to replace critical thinking
16. name_calling — Labels opponents with negative terms
17. flag_waving — Exploits patriotism to justify a position
18. thought_terminating_cliche — Uses clichés to shut down debate

## Input
TEXT: {text}
LANGUAGE: {language}

## Required JSON Output
Return ONLY valid JSON (no markdown fences, no extra text):
{{
  "is_persuasive": true,
  "techniques": ["loaded_language", "appeal_to_fear"],
  "technique_confidences": {{"loaded_language": 0.92, "appeal_to_fear": 0.87}},
  "highlighted_phrases": [
    {{
      "phrase": "exact substring from input text",
      "technique": "loaded_language",
      "evidence": "The word 'catastrophic' is emotionally charged, designed to trigger fear"
    }}
  ],
  "explanation_en": "Detailed 2-3 sentence explanation in English with specific evidence from the text",
  "explanation_fr": "Same explanation in French",
  "explanation_ar": "Same explanation in Arabic",
  "neutral_rewrite": "Factual rewrite removing all manipulation, same language as input",
  "severity": 3,
  "manipulation_score": 72
}}

Rules:
- severity: integer 0-5. Use 0 if not persuasive.
- manipulation_score: integer 0-100. Use 0 if not persuasive.
- highlighted_phrases: EXACT substrings from the input text. Each phrase must appear verbatim.
- Only detect techniques with high confidence. Quality over quantity.
- If the text is neutral, set is_persuasive=false, techniques=[], highlighted_phrases=[], severity=0, manipulation_score=0.
- Explanations must cite specific phrases as evidence.
"""


@dataclass
class AnalysisResult:
    """Container for the full analysis of a single text."""

    text: str
    language: str
    is_persuasive: bool
    techniques: list[str]
    technique_probabilities: dict[str, float]
    explanation: str
    neutral_rewrite: str
    severity_score: float
    manipulation_score: float
    cross_lingual_explanations: dict[str, str] = field(default_factory=dict)
    highlighted_phrases: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "is_persuasive": self.is_persuasive,
            "techniques": self.techniques,
            "technique_probabilities": self.technique_probabilities,
            "explanation": self.explanation,
            "neutral_rewrite": self.neutral_rewrite,
            "severity_score": self.severity_score,
            "manipulation_score": self.manipulation_score,
            "cross_lingual_explanations": self.cross_lingual_explanations,
            "highlighted_phrases": self.highlighted_phrases,
        }


class PersuasixPipeline:
    """
    Orchestrate the full PersuasiX analysis pipeline.

    Primary: OpenAI GPT-4o-mini (accurate, production-quality results)
    Fallback: Local ML models (RoBERTa + FLAN-T5, requires fine-tuning for good results)
    """

    def __init__(
        self,
        detector: PersuasionDetector | None = None,
        explainer: PersuasionExplainer | None = None,
        neutralizer: TextNeutralizer | None = None,
        scorer: ReliabilityScorer | None = None,
        device: str | None = None,
        detection_threshold: float = 0.5,
        openai_client: Any = None,
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_threshold = detection_threshold

        self.detector = detector
        self.explainer = explainer
        self.neutralizer = neutralizer
        self.scorer = scorer

        self._openai = openai_client
        self._openai_model = openai_model

    @classmethod
    def from_pretrained(
        cls,
        detector_path: str | None = None,
        explainer_path: str | None = None,
        neutralizer_path: str | None = None,
        device: str | None = None,
    ) -> "PersuasixPipeline":
        """Load a pipeline from pretrained checkpoints."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        detector = None
        if detector_path:
            detector = PersuasionDetector.from_pretrained(detector_path)
            detector.to(device)

        explainer = None
        if explainer_path:
            explainer = PersuasionExplainer.load(explainer_path, device=device)

        neutralizer = None
        if neutralizer_path:
            neutralizer = TextNeutralizer.load(neutralizer_path, device=device)

        scorer = ReliabilityScorer(device=device)

        return cls(
            detector=detector,
            explainer=explainer,
            neutralizer=neutralizer,
            scorer=scorer,
            device=device,
        )

    @classmethod
    def from_default_models(cls, device: str | None = None) -> "PersuasixPipeline":
        """Create a pipeline — uses OpenAI if available, otherwise local models."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        openai_client = None
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key.startswith("sk-"):
                from openai import OpenAI

                openai_client = OpenAI()
                logger.info("OpenAI client ready — using GPT-4o-mini for analysis")
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")

        detector = None
        explainer = None
        neutralizer = None
        scorer = None

        if not openai_client:
            logger.info("Loading local ML models (not fine-tuned — results may be inaccurate) …")
            detector = PersuasionDetector(model_name="roberta-base", num_labels=18)
            detector.to(device)
            explainer = PersuasionExplainer(model_name="google/flan-t5-base", device=device)
            neutralizer = TextNeutralizer(model_name="google/flan-t5-base", device=device)
            scorer = ReliabilityScorer(device=device)

        return cls(
            detector=detector,
            explainer=explainer,
            neutralizer=neutralizer,
            scorer=scorer,
            device=device,
            openai_client=openai_client,
        )

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str, language: str = "en") -> AnalysisResult:
        """Run the full pipeline on a single text."""
        if self._openai:
            try:
                return self._analyze_openai(text, language)
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")
                if self.detector:
                    logger.info("Falling back to local models …")
                    return self._analyze_local(text, language)
                raise

        return self._analyze_local(text, language)

    def analyze_batch(self, texts: list[str], language: str = "en") -> list[AnalysisResult]:
        """Analyze multiple texts."""
        return [self.analyze(text, language) for text in texts]

    def fact_check(self, text: str, language: str = "en") -> dict:
        """Run fact-checking on the text and return a report."""
        if not HAS_FACT_CHECKER:
            return {"error": "Fact checker module not available"}
        try:
            checker = PersuasixFactChecker()
            report = checker.check(text, language=language)
            return report.to_dict()
        except Exception as e:
            logger.error(f"Fact check failed: {e}")
            return {"error": str(e)}

    def detect_spans(self, text: str) -> dict:
        """Run span-level detection to find exact manipulative phrases."""
        if not HAS_SPAN_DETECTOR:
            return {"error": "Span detector module not available"}
        try:
            detector = HybridSpanDetector(use_llm=self._openai is not None)
            result = detector.detect(text)
            return result.to_dict()
        except Exception as e:
            logger.error(f"Span detection failed: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # OpenAI-powered analysis
    # ------------------------------------------------------------------

    def _analyze_openai(self, text: str, language: str) -> AnalysisResult:
        lang_names = {
            "en": "English", "fr": "French", "ar": "Arabic",
            "es": "Spanish", "de": "German", "zh": "Chinese",
            "hi": "Hindi",
        }
        prompt = ANALYSIS_PROMPT.format(
            text=text,
            language=lang_names.get(language, "English"),
        )

        response = self._openai.chat.completions.create(
            model=self._openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        content = response.choices[0].message.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(content)

        techniques = [t for t in data.get("techniques", []) if t in TECHNIQUE_LABELS]
        confidences = data.get("technique_confidences", {})

        prob_dict = {}
        for t in TECHNIQUE_LABELS:
            if t in techniques:
                prob_dict[t] = round(float(confidences.get(t, 0.9)), 4)
            else:
                prob_dict[t] = round(float(confidences.get(t, 0.0)), 4)

        lang_key = f"explanation_{language}"
        explanation = data.get(lang_key, data.get("explanation_en", ""))

        cross_lingual = {}
        for lang in ["en", "fr", "ar", "es", "de", "zh", "hi"]:
            key = f"explanation_{lang}"
            if key in data and data[key]:
                cross_lingual[lang] = data[key]

        severity = min(float(data.get("severity", 0)), 5.0)
        manip = min(float(data.get("manipulation_score", 0)) / 100.0, 1.0)

        highlighted = []
        for h in data.get("highlighted_phrases", []):
            phrase = h.get("phrase", "")
            if phrase and phrase in text:
                highlighted.append({
                    "phrase": phrase,
                    "technique": h.get("technique", ""),
                    "evidence": h.get("evidence", ""),
                })

        return AnalysisResult(
            text=text,
            language=language,
            is_persuasive=data.get("is_persuasive", len(techniques) > 0),
            techniques=techniques,
            technique_probabilities=prob_dict,
            explanation=explanation,
            neutral_rewrite=data.get("neutral_rewrite", text),
            severity_score=round(severity, 2),
            manipulation_score=round(manip, 4),
            cross_lingual_explanations=cross_lingual,
            highlighted_phrases=highlighted,
        )

    # ------------------------------------------------------------------
    # Local ML pipeline (fallback)
    # ------------------------------------------------------------------

    def _analyze_local(self, text: str, language: str = "en") -> AnalysisResult:
        techniques, probs = self._detect(text)
        is_persuasive = len(techniques) > 0

        explanation = ""
        neutral_rewrite = text
        manipulation_score = 0.0
        cross_lingual: dict[str, str] = {}

        if is_persuasive:
            if self.explainer:
                explanation = self.explainer.explain(text, techniques, language)
                for lang in ["en", "fr", "ar"]:
                    if lang != language:
                        cross_lingual[lang] = self.explainer.explain(text, techniques, lang)
                    cross_lingual[language] = explanation

            if self.neutralizer:
                neutral_rewrite = self.neutralizer.neutralize(text, techniques)

            if self.scorer:
                score_result = self.scorer.score_manipulation(text, neutral_rewrite)
                manipulation_score = score_result["manipulation_score"]

        severity = min(len(techniques) * manipulation_score * 5, 5.0) if is_persuasive else 0.0

        return AnalysisResult(
            text=text,
            language=language,
            is_persuasive=is_persuasive,
            techniques=techniques,
            technique_probabilities=probs,
            explanation=explanation,
            neutral_rewrite=neutral_rewrite,
            severity_score=round(severity, 2),
            manipulation_score=round(manipulation_score, 4),
            cross_lingual_explanations=cross_lingual,
            highlighted_phrases=[],
        )

    def _detect(self, text: str) -> tuple[list[str], dict[str, float]]:
        if self.detector is None:
            return [], {}

        tokenizer = PersuasionDetector.get_tokenizer()
        encoding = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        result = self.detector.predict(
            input_ids,
            attention_mask,
            threshold=self.detection_threshold,
        )

        probs = result["probabilities"][0].cpu().tolist()
        preds = result["predictions"][0].cpu().tolist()

        prob_dict = {
            TECHNIQUE_LABELS[i]: round(probs[i], 4)
            for i in range(len(TECHNIQUE_LABELS))
        }

        techniques = [
            TECHNIQUE_LABELS[i]
            for i in range(len(TECHNIQUE_LABELS))
            if preds[i] == 1
        ]

        return techniques, prob_dict
