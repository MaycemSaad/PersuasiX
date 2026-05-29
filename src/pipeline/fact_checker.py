"""
Fact-checking API integration for PersuasiX.

Integrates with multiple fact-checking services to verify claims
found in analyzed texts, adding a credibility layer to persuasion detection.

Supported APIs:
  1. Google Fact Check Tools API (free, requires API key)
  2. ClaimBuster API (free for research, claim detection scoring)
  3. OpenAI-powered claim extraction + verification

Pipeline:
  Text → Claim Extraction → Fact Check Lookup → Credibility Scoring → Report
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedClaim:
    """A factual claim extracted from text."""
    text: str               # The claim text
    start: int = 0          # Character offset in source
    end: int = 0            # Character offset end
    claim_type: str = ""    # statistical, causal, authority, prediction, etc.
    checkworthiness: float = 0.0  # 0-1 score of how checkworthy this claim is

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "claim_type": self.claim_type,
            "checkworthiness": round(self.checkworthiness, 3),
        }


@dataclass
class FactCheckResult:
    """Result from fact-checking a single claim."""
    claim: str
    verdict: str            # true, false, mostly_true, mostly_false, mixed, unverified
    confidence: float       # 0-1 confidence in the verdict
    source: str             # Which API/method provided this
    source_url: str = ""    # URL to the fact-check article
    publisher: str = ""     # Who published the fact-check
    review_date: str = ""   # When the fact-check was published
    explanation: str = ""   # Brief explanation

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "source_url": self.source_url,
            "publisher": self.publisher,
            "review_date": self.review_date,
            "explanation": self.explanation,
        }


@dataclass
class FactCheckReport:
    """Complete fact-checking report for a text."""
    text: str
    claims: list[ExtractedClaim] = field(default_factory=list)
    results: list[FactCheckResult] = field(default_factory=list)
    credibility_score: float = 0.0   # 0-100 overall credibility
    num_verified: int = 0
    num_false: int = 0
    num_unverified: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "claims": [c.to_dict() for c in self.claims],
            "results": [r.to_dict() for r in self.results],
            "credibility_score": round(self.credibility_score, 1),
            "num_claims": len(self.claims),
            "num_verified": self.num_verified,
            "num_false": self.num_false,
            "num_unverified": self.num_unverified,
        }


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_PROMPT = """\
You are an expert fact-checker. Extract all verifiable factual claims from the following text.

A "claim" is a statement that can be objectively verified as true or false. Skip:
- Opinions and subjective statements
- Questions
- Future predictions that can't be verified yet
- Vague statements without specific facts

For each claim, classify its type:
- statistical: Contains numbers, percentages, data
- causal: Claims X causes Y
- authority: Attributes a claim to a person/organization
- historical: References past events
- scientific: References research or scientific findings
- comparative: Compares two things

TEXT:
{text}

Return ONLY valid JSON:
{{
  "claims": [
    {{
      "text": "The exact claim as it appears in the text",
      "claim_type": "statistical",
      "checkworthiness": 0.85
    }}
  ]
}}
"""


class ClaimExtractor:
    """Extract checkworthy claims from text."""

    def __init__(self, method: str = "llm") -> None:
        self.method = method
        self._openai = None
        if method == "llm":
            self._init_openai()

    def _init_openai(self) -> None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key.startswith("sk-"):
                from openai import OpenAI
                self._openai = OpenAI()
        except Exception:
            pass

    def extract(self, text: str) -> list[ExtractedClaim]:
        """Extract claims from text."""
        if self.method == "llm" and self._openai:
            return self._extract_llm(text)
        return self._extract_heuristic(text)

    def _extract_llm(self, text: str) -> list[ExtractedClaim]:
        """Use LLM to extract claims."""
        try:
            prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:3000])
            response = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()
            content = content.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(content)

            claims = []
            for c in data.get("claims", []):
                claim_text = c.get("text", "")
                idx = text.find(claim_text)
                claims.append(ExtractedClaim(
                    text=claim_text,
                    start=max(idx, 0),
                    end=max(idx, 0) + len(claim_text),
                    claim_type=c.get("claim_type", ""),
                    checkworthiness=float(c.get("checkworthiness", 0.5)),
                ))
            return claims

        except Exception as e:
            logger.warning(f"LLM claim extraction failed: {e}")
            return self._extract_heuristic(text)

    @staticmethod
    def _extract_heuristic(text: str) -> list[ExtractedClaim]:
        """Rule-based claim extraction (fallback)."""
        claims: list[ExtractedClaim] = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Patterns that indicate checkworthy claims
        patterns = [
            (r'\d+(?:\.\d+)?%', "statistical", 0.8),
            (r'(?:according to|as stated by|said|confirmed|reported)', "authority", 0.7),
            (r'(?:study|research|survey|data|evidence) (?:shows?|found|indicates?|suggests?)', "scientific", 0.9),
            (r'(?:cause[sd]?|leads? to|results? in|because of)', "causal", 0.6),
            (r'(?:increased?|decreased?|grew|fell|rose|dropped) by', "statistical", 0.7),
            (r'(?:more|less|higher|lower|better|worse) than', "comparative", 0.5),
            (r'(?:in \d{4}|last (?:year|month|week|decade))', "historical", 0.6),
            (r'(?:million|billion|trillion|thousand)', "statistical", 0.7),
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            best_score = 0.0
            best_type = "general"

            for pattern, claim_type, score in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    if score > best_score:
                        best_score = score
                        best_type = claim_type

            if best_score >= 0.5:
                idx = text.find(sentence)
                claims.append(ExtractedClaim(
                    text=sentence,
                    start=max(idx, 0),
                    end=max(idx, 0) + len(sentence),
                    claim_type=best_type,
                    checkworthiness=best_score,
                ))

        return claims


# ---------------------------------------------------------------------------
# Google Fact Check Tools API
# ---------------------------------------------------------------------------

class GoogleFactChecker:
    """Integration with Google Fact Check Tools API."""

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self) -> None:
        self.api_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
        if not self.api_key:
            logger.info("Google Fact Check API key not set (GOOGLE_FACTCHECK_API_KEY)")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def check(self, claim: str, language: str = "en") -> list[FactCheckResult]:
        """Search for fact-checks matching a claim."""
        if not self.api_key:
            return []

        try:
            params = {
                "key": self.api_key,
                "query": claim[:200],
                "languageCode": language,
            }
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results: list[FactCheckResult] = []
            for item in data.get("claims", [])[:3]:
                for review in item.get("claimReview", []):
                    rating = review.get("textualRating", "").lower()
                    verdict = self._normalize_rating(rating)

                    results.append(FactCheckResult(
                        claim=item.get("text", claim),
                        verdict=verdict,
                        confidence=0.85,
                        source="google_factcheck",
                        source_url=review.get("url", ""),
                        publisher=review.get("publisher", {}).get("name", ""),
                        review_date=review.get("reviewDate", ""),
                        explanation=f"Rated '{rating}' by {review.get('publisher', {}).get('name', 'unknown')}",
                    ))

            return results

        except Exception as e:
            logger.warning(f"Google Fact Check API error: {e}")
            return []

    @staticmethod
    def _normalize_rating(rating: str) -> str:
        """Normalize various fact-check ratings to our standard labels."""
        rating = rating.lower().strip()
        if any(w in rating for w in ["true", "correct", "accurate", "vrai"]):
            if any(w in rating for w in ["mostly", "partly", "half"]):
                return "mostly_true"
            return "true"
        elif any(w in rating for w in ["false", "wrong", "incorrect", "faux", "pants on fire"]):
            if any(w in rating for w in ["mostly", "partly"]):
                return "mostly_false"
            return "false"
        elif any(w in rating for w in ["mixed", "half", "misleading"]):
            return "mixed"
        return "unverified"


# ---------------------------------------------------------------------------
# ClaimBuster API
# ---------------------------------------------------------------------------

class ClaimBusterChecker:
    """Integration with ClaimBuster API for claim detection scoring."""

    BASE_URL = "https://idir.uta.edu/claimbuster/api/v2"

    def __init__(self) -> None:
        self.api_key = os.environ.get("CLAIMBUSTER_API_KEY", "")
        if not self.api_key:
            logger.info("ClaimBuster API key not set (CLAIMBUSTER_API_KEY)")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def score_claims(self, text: str) -> list[dict]:
        """Score sentences for check-worthiness."""
        if not self.api_key:
            return []

        try:
            url = f"{self.BASE_URL}/score/text/"
            headers = {"x-api-key": self.api_key}
            payload = {"input_text": text[:5000]}

            response = requests.post(url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "text": item.get("text", ""),
                    "score": float(item.get("score", 0)),
                })

            return sorted(results, key=lambda x: -x["score"])

        except Exception as e:
            logger.warning(f"ClaimBuster API error: {e}")
            return []


# ---------------------------------------------------------------------------
# LLM-powered fact verification
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """\
You are an expert fact-checker. Assess the truthfulness of the following claim.

CLAIM: "{claim}"
CONTEXT: This claim was found in a text being analyzed for persuasion techniques.

Evaluate the claim based on your knowledge. Rate it as:
- "true": Verified accurate based on widely accepted evidence
- "mostly_true": Largely accurate with minor inaccuracies
- "mixed": Contains both accurate and inaccurate elements
- "mostly_false": Largely inaccurate with some truth
- "false": Demonstrably incorrect
- "unverifiable": Cannot be determined without more context or data

Return ONLY valid JSON:
{{
  "verdict": "true",
  "confidence": 0.85,
  "explanation": "Brief explanation with reasoning..."
}}
"""


class LLMFactChecker:
    """Use LLM to verify claims when APIs don't have results."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = None
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key.startswith("sk-"):
                from openai import OpenAI
                self._client = OpenAI()
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def verify(self, claim: str) -> FactCheckResult | None:
        """Verify a claim using LLM knowledge."""
        if not self._client:
            return None

        try:
            prompt = VERIFY_PROMPT.format(claim=claim)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()
            content = content.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(content)

            return FactCheckResult(
                claim=claim,
                verdict=data.get("verdict", "unverifiable"),
                confidence=float(data.get("confidence", 0.5)) * 0.8,  # discount LLM confidence
                source="llm_verification",
                explanation=data.get("explanation", ""),
            )

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Unified Fact Checker
# ---------------------------------------------------------------------------

class PersuasixFactChecker:
    """
    Unified fact-checking pipeline for PersuasiX.

    Pipeline:
    1. Extract checkworthy claims from text
    2. Search Google Fact Check Tools for existing fact-checks
    3. Score with ClaimBuster if available
    4. Verify remaining claims with LLM
    5. Compute overall credibility score
    """

    def __init__(self) -> None:
        self.claim_extractor = ClaimExtractor(method="llm")
        self.google_checker = GoogleFactChecker()
        self.claimbuster = ClaimBusterChecker()
        self.llm_checker = LLMFactChecker()

        available = []
        if self.google_checker.is_available:
            available.append("Google Fact Check")
        if self.claimbuster.is_available:
            available.append("ClaimBuster")
        if self.llm_checker.is_available:
            available.append("LLM Verification")

        logger.info(f"FactChecker initialized with: {', '.join(available) or 'heuristic only'}")

    def check(self, text: str, language: str = "en") -> FactCheckReport:
        """Run the full fact-checking pipeline on a text."""
        # Step 1: Extract claims
        claims = self.claim_extractor.extract(text)
        if not claims:
            return FactCheckReport(text=text, credibility_score=50.0)

        # Sort by checkworthiness and limit
        claims.sort(key=lambda c: -c.checkworthiness)
        claims = claims[:10]  # Max 10 claims

        all_results: list[FactCheckResult] = []

        # Step 2: Check each claim
        for claim in claims:
            results = self._check_single_claim(claim, language)
            all_results.extend(results)

        # Step 3: Compute credibility score
        report = self._build_report(text, claims, all_results)
        return report

    def _check_single_claim(
        self,
        claim: ExtractedClaim,
        language: str,
    ) -> list[FactCheckResult]:
        """Check a single claim against all available sources."""
        results: list[FactCheckResult] = []

        # Try Google Fact Check
        if self.google_checker.is_available:
            google_results = self.google_checker.check(claim.text, language)
            results.extend(google_results)

        # If no results from APIs, try LLM
        if not results and self.llm_checker.is_available:
            llm_result = self.llm_checker.verify(claim.text)
            if llm_result:
                results.append(llm_result)

        # If still nothing, mark as unverified
        if not results:
            results.append(FactCheckResult(
                claim=claim.text,
                verdict="unverified",
                confidence=0.0,
                source="none",
                explanation="No fact-checking source available for this claim.",
            ))

        return results

    def _build_report(
        self,
        text: str,
        claims: list[ExtractedClaim],
        results: list[FactCheckResult],
    ) -> FactCheckReport:
        """Build the final fact-check report with credibility scoring."""
        num_verified = 0
        num_false = 0
        num_unverified = 0

        verdict_scores = {
            "true": 1.0,
            "mostly_true": 0.75,
            "mixed": 0.5,
            "mostly_false": 0.25,
            "false": 0.0,
            "unverifiable": 0.5,
            "unverified": 0.5,
        }

        claim_scores: list[float] = []
        for r in results:
            score = verdict_scores.get(r.verdict, 0.5)
            claim_scores.append(score * r.confidence if r.confidence > 0 else 0.5)

            if r.verdict in ("true", "mostly_true"):
                num_verified += 1
            elif r.verdict in ("false", "mostly_false"):
                num_false += 1
            else:
                num_unverified += 1

        if claim_scores:
            avg_score = sum(claim_scores) / len(claim_scores)
            credibility = avg_score * 100
        else:
            credibility = 50.0

        return FactCheckReport(
            text=text,
            claims=claims,
            results=results,
            credibility_score=round(credibility, 1),
            num_verified=num_verified,
            num_false=num_false,
            num_unverified=num_unverified,
        )
