"""
Span-level persuasion technique detection.

Instead of just classifying entire texts, this module identifies
the EXACT character spans that contain manipulative language,
along with the specific technique used at each span.

Architecture:
  - Token-level NER-style detection using RoBERTa + CRF head
  - BIO tagging scheme: B-technique, I-technique, O
  - Post-processing: merge contiguous spans, filter by confidence

Supports:
  1. Model-based span detection (fine-tuned token classifier)
  2. LLM-based span detection (GPT-4o-mini with structured prompts)
  3. Hybrid: model proposes candidates, LLM validates & explains
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedSpan:
    """A single detected manipulative span in text."""
    start: int          # Character offset start
    end: int            # Character offset end
    text: str           # The exact substring
    technique: str      # Technique label (e.g., "loaded_language")
    confidence: float   # Model confidence [0, 1]
    evidence: str = ""  # Optional explanation of why this span is manipulative

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "technique": self.technique,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


@dataclass
class SpanDetectionResult:
    """Full span detection result for a text."""
    text: str
    spans: list[DetectedSpan] = field(default_factory=list)
    techniques_found: list[str] = field(default_factory=list)
    coverage: float = 0.0  # Fraction of text covered by spans

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
            "techniques_found": self.techniques_found,
            "coverage": round(self.coverage, 4),
            "num_spans": len(self.spans),
        }


# ---------------------------------------------------------------------------
# Technique labels for BIO tagging
# ---------------------------------------------------------------------------

TECHNIQUE_LABELS = [
    "appeal_to_fear", "appeal_to_authority", "bandwagon", "false_dilemma",
    "ad_hominem", "straw_man", "red_herring", "loaded_language",
    "whataboutism", "causal_oversimplification", "appeal_to_emotion",
    "repetition", "exaggeration", "doubt", "slogans", "name_calling",
    "flag_waving", "thought_terminating_cliche",
]

# BIO tag set: O + B-tech + I-tech for each technique
BIO_LABELS = ["O"]
for tech in TECHNIQUE_LABELS:
    BIO_LABELS.append(f"B-{tech}")
    BIO_LABELS.append(f"I-{tech}")

LABEL_TO_ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
NUM_LABELS = len(BIO_LABELS)


# ---------------------------------------------------------------------------
# Model: RoBERTa + CRF for span detection
# ---------------------------------------------------------------------------

class CRFLayer(nn.Module):
    """Conditional Random Field layer for sequence labeling."""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss."""
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        return (forward_score - gold_score).mean()

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding to find best tag sequence."""
        batch_size, seq_len, _ = emissions.shape
        best_paths: list[list[int]] = []

        for i in range(batch_size):
            length = int(mask[i].sum().item())
            emit = emissions[i, :length]
            path = self._viterbi_decode(emit)
            path = path + [0] * (seq_len - length)
            best_paths.append(path)

        return best_paths

    def _score_sentence(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            m = mask[:, t].float()
            emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit_score + trans_score) * m

        last_tag_idx = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _forward_algorithm(
        self, emissions: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape

        alphas = self.start_transitions + emissions[:, 0]

        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            alpha_expand = alphas.unsqueeze(2)
            scores = alpha_expand + trans + emit
            new_alphas = torch.logsumexp(scores, dim=1)
            m = mask[:, t].unsqueeze(1).float()
            alphas = new_alphas * m + alphas * (1 - m)

        alphas += self.end_transitions
        return torch.logsumexp(alphas, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor) -> list[int]:
        seq_len, num_tags = emissions.shape

        viterbi = self.start_transitions + emissions[0]
        backpointers: list[torch.Tensor] = []

        for t in range(1, seq_len):
            scores = viterbi.unsqueeze(1) + self.transitions
            max_scores, bp = scores.max(dim=0)
            viterbi = max_scores + emissions[t]
            backpointers.append(bp)

        viterbi += self.end_transitions
        best_last = viterbi.argmax().item()

        best_path = [best_last]
        for bp in reversed(backpointers):
            best_path.append(bp[best_path[-1]].item())
        best_path.reverse()

        return best_path


class SpanDetectionModel(nn.Module):
    """
    RoBERTa + Linear + CRF model for token-level span detection.

    Architecture:
        RoBERTa encoder → per-token hidden states → Linear(hidden, num_tags)
        → CRF layer for structured prediction
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_tags: int = NUM_LABELS,
        dropout: float = 0.1,
        use_crf: bool = True,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.use_crf = use_crf

        if HAS_TRANSFORMERS:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
            hidden_size = self.config.hidden_size
        else:
            hidden_size = 768
            self.encoder = None

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)

        if use_crf:
            self.crf = CRFLayer(num_tags)
        else:
            self.crf = None
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.encoder is None:
            raise RuntimeError("Transformers not available")

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        result: dict[str, Any] = {"emissions": emissions}

        if labels is not None:
            if self.use_crf:
                loss = self.crf(emissions, labels, attention_mask.bool())
                result["loss"] = loss
            else:
                loss = self.loss_fn(
                    emissions.view(-1, self.num_tags),
                    labels.view(-1),
                )
                result["loss"] = loss

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        """Predict tag sequences."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            emissions = outputs["emissions"]

            if self.use_crf:
                tag_sequences = self.crf.decode(emissions, attention_mask.bool())
            else:
                tag_sequences = emissions.argmax(dim=-1).tolist()

        return tag_sequences

    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.state_dict(),
            "num_tags": self.num_tags,
            "use_crf": self.use_crf,
        }, path)

    @classmethod
    def load(cls, path: str, model_name: str = "roberta-base") -> "SpanDetectionModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            model_name=model_name,
            num_tags=checkpoint["num_tags"],
            use_crf=checkpoint.get("use_crf", True),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


# ---------------------------------------------------------------------------
# LLM-based span detector
# ---------------------------------------------------------------------------

SPAN_DETECTION_PROMPT = """\
You are an expert in rhetoric and persuasion analysis. Given a text, identify \
the EXACT character-level spans that contain manipulative or persuasive language.

For each span, provide:
1. The exact substring (must appear verbatim in the text)
2. The start character offset
3. The end character offset
4. The specific technique used
5. A brief explanation of why it's manipulative

## Technique Catalog
{techniques}

## Text to analyze
"{text}"

## Required JSON Output
Return ONLY valid JSON (no markdown):
{{
  "spans": [
    {{
      "start": 0,
      "end": 25,
      "text": "exact substring from input",
      "technique": "loaded_language",
      "confidence": 0.92,
      "evidence": "Brief explanation..."
    }}
  ]
}}

Rules:
- "text" must be an EXACT substring of the input
- "start" and "end" must be correct character offsets
- Only flag spans you are confident about (confidence > 0.7)
- Multiple spans can overlap or be nested
- If no manipulative spans found, return {{"spans": []}}
"""


class LLMSpanDetector:
    """Use GPT-4o-mini for span-level detection."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key.startswith("sk-"):
                from openai import OpenAI
                self._client = OpenAI()
        except Exception as e:
            logger.warning(f"OpenAI not available for span detection: {e}")

    def detect(self, text: str) -> SpanDetectionResult:
        """Detect manipulative spans in text."""
        if not self._client:
            return SpanDetectionResult(text=text)

        techniques_str = "\n".join(
            f"- {tech}: {tech.replace('_', ' ').title()}"
            for tech in TECHNIQUE_LABELS
        )

        prompt = SPAN_DETECTION_PROMPT.format(
            text=text,
            techniques=techniques_str,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()
            content = content.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(content)

            spans = []
            for s in data.get("spans", []):
                span_text = s.get("text", "")
                # Verify span exists in text
                idx = text.find(span_text)
                if idx >= 0:
                    spans.append(DetectedSpan(
                        start=idx,
                        end=idx + len(span_text),
                        text=span_text,
                        technique=s.get("technique", ""),
                        confidence=float(s.get("confidence", 0.8)),
                        evidence=s.get("evidence", ""),
                    ))

            techniques_found = list(set(s.technique for s in spans))
            total_span_chars = sum(s.end - s.start for s in spans)
            coverage = total_span_chars / max(len(text), 1)

            return SpanDetectionResult(
                text=text,
                spans=spans,
                techniques_found=techniques_found,
                coverage=min(coverage, 1.0),
            )

        except Exception as e:
            logger.error(f"LLM span detection failed: {e}")
            return SpanDetectionResult(text=text)


# ---------------------------------------------------------------------------
# Hybrid span detector
# ---------------------------------------------------------------------------

class HybridSpanDetector:
    """
    Combine model-based and LLM-based span detection.

    Strategy:
    1. Model proposes candidate spans (fast)
    2. LLM validates and enriches with explanations (accurate)
    3. Merge results with confidence-weighted scoring
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_llm: bool = True,
        llm_model: str = "gpt-4o-mini",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold

        # Model-based detector
        self._model = None
        self._tokenizer = None
        if model_path and HAS_TRANSFORMERS:
            try:
                self._model = SpanDetectionModel.load(model_path)
                self._model.eval()
                self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                logger.info(f"Span detection model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load span model: {e}")

        # LLM-based detector
        self._llm = None
        if use_llm:
            self._llm = LLMSpanDetector(model=llm_model)

    def detect(self, text: str) -> SpanDetectionResult:
        """Run hybrid span detection."""
        model_spans: list[DetectedSpan] = []
        llm_spans: list[DetectedSpan] = []

        # Phase 1: Model-based detection
        if self._model and self._tokenizer:
            model_spans = self._model_detect(text)

        # Phase 2: LLM-based detection
        if self._llm:
            llm_result = self._llm.detect(text)
            llm_spans = llm_result.spans

        # Phase 3: Merge
        merged = self._merge_spans(model_spans, llm_spans)

        # Filter by confidence
        filtered = [s for s in merged if s.confidence >= self.confidence_threshold]

        techniques_found = list(set(s.technique for s in filtered))
        total_span_chars = sum(s.end - s.start for s in filtered)
        coverage = total_span_chars / max(len(text), 1)

        return SpanDetectionResult(
            text=text,
            spans=filtered,
            techniques_found=techniques_found,
            coverage=min(coverage, 1.0),
        )

    def _model_detect(self, text: str) -> list[DetectedSpan]:
        """Run model-based span detection."""
        encoding = self._tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping").squeeze(0)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        tag_sequence = self._model.predict(input_ids, attention_mask)[0]

        spans: list[DetectedSpan] = []
        current_span = None

        for idx, tag_id in enumerate(tag_sequence):
            if idx >= len(offset_mapping):
                break

            token_start, token_end = offset_mapping[idx].tolist()
            if token_start == 0 and token_end == 0:
                continue

            label = ID_TO_LABEL.get(tag_id, "O")

            if label.startswith("B-"):
                if current_span:
                    spans.append(current_span)
                technique = label[2:]
                current_span = DetectedSpan(
                    start=int(token_start),
                    end=int(token_end),
                    text=text[int(token_start):int(token_end)],
                    technique=technique,
                    confidence=0.7,
                )
            elif label.startswith("I-") and current_span:
                technique = label[2:]
                if technique == current_span.technique:
                    current_span.end = int(token_end)
                    current_span.text = text[current_span.start:current_span.end]
                else:
                    spans.append(current_span)
                    current_span = None
            else:
                if current_span:
                    spans.append(current_span)
                    current_span = None

        if current_span:
            spans.append(current_span)

        return spans

    @staticmethod
    def _merge_spans(
        model_spans: list[DetectedSpan],
        llm_spans: list[DetectedSpan],
    ) -> list[DetectedSpan]:
        """Merge spans from model and LLM, boosting confidence for agreed spans."""
        if not model_spans:
            return llm_spans
        if not llm_spans:
            return model_spans

        merged: list[DetectedSpan] = []
        used_llm: set[int] = set()

        for ms in model_spans:
            best_overlap = 0.0
            best_llm_idx = -1

            for i, ls in enumerate(llm_spans):
                if i in used_llm:
                    continue
                overlap = _span_overlap(ms, ls)
                if overlap > best_overlap and ms.technique == ls.technique:
                    best_overlap = overlap
                    best_llm_idx = i

            if best_overlap > 0.3 and best_llm_idx >= 0:
                ls = llm_spans[best_llm_idx]
                used_llm.add(best_llm_idx)
                merged.append(DetectedSpan(
                    start=min(ms.start, ls.start),
                    end=max(ms.end, ls.end),
                    text=ms.text if len(ms.text) > len(ls.text) else ls.text,
                    technique=ms.technique,
                    confidence=min(1.0, (ms.confidence + ls.confidence) / 2 + 0.15),
                    evidence=ls.evidence or ms.evidence,
                ))
            else:
                merged.append(ms)

        for i, ls in enumerate(llm_spans):
            if i not in used_llm:
                merged.append(ls)

        merged.sort(key=lambda s: s.start)
        return merged


def _span_overlap(a: DetectedSpan, b: DetectedSpan) -> float:
    """Compute overlap ratio between two spans."""
    overlap_start = max(a.start, b.start)
    overlap_end = min(a.end, b.end)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_len = overlap_end - overlap_start
    union_len = max(a.end, b.end) - min(a.start, b.start)
    return overlap_len / max(union_len, 1)
