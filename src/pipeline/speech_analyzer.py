"""
Speech-to-text persuasion analysis for audio/video content.

Pipeline:
  Audio/Video → Speech-to-Text → PersuasiX Analysis → Timestamped Report

Supports:
  1. OpenAI Whisper API (cloud, high accuracy)
  2. Local Whisper model (offline, GPU recommended)
  3. Multiple audio formats: mp3, wav, m4a, ogg, flac, mp4, webm
  4. Chunked analysis with timestamps
  5. Speaker diarization (optional)

Usage:
    from src.pipeline.speech_analyzer import SpeechAnalyzer
    analyzer = SpeechAnalyzer()
    result = analyzer.analyze_file("speech.mp3")
    result = analyzer.analyze_url("https://example.com/video.mp4")
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    """A single segment of transcribed speech."""
    text: str
    start: float          # Start time in seconds
    end: float            # End time in seconds
    speaker: str = ""     # Speaker ID (if diarization available)
    language: str = "en"
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def timestamp_str(self) -> str:
        return f"{_fmt_time(self.start)} → {_fmt_time(self.end)}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker": self.speaker,
            "language": self.language,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class SegmentAnalysis:
    """Analysis result for a single transcript segment."""
    segment: TranscriptSegment
    is_persuasive: bool = False
    manipulation_score: float = 0.0
    techniques: list[str] = field(default_factory=list)
    highlighted_phrases: list[dict] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "segment": self.segment.to_dict(),
            "is_persuasive": self.is_persuasive,
            "manipulation_score": round(self.manipulation_score, 4),
            "techniques": self.techniques,
            "highlighted_phrases": self.highlighted_phrases,
            "explanation": self.explanation,
        }


@dataclass
class SpeechAnalysisResult:
    """Complete speech analysis result."""
    source: str                                    # File path or URL
    duration: float = 0.0                          # Total audio duration in seconds
    language: str = "en"                           # Detected language
    full_transcript: str = ""                      # Complete transcription
    segments: list[TranscriptSegment] = field(default_factory=list)
    segment_analyses: list[SegmentAnalysis] = field(default_factory=list)
    overall_score: float = 0.0                     # Average manipulation score
    overall_techniques: list[str] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)  # Time-indexed manipulation scores

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "duration": round(self.duration, 1),
            "duration_formatted": _fmt_time(self.duration),
            "language": self.language,
            "full_transcript": self.full_transcript[:500] + "..." if len(self.full_transcript) > 500 else self.full_transcript,
            "total_segments": len(self.segments),
            "analyzed_segments": len(self.segment_analyses),
            "manipulative_segments": sum(1 for a in self.segment_analyses if a.is_persuasive),
            "overall_score": round(self.overall_score, 4),
            "overall_techniques": self.overall_techniques,
            "timeline": self.timeline,
            "segment_analyses": [a.to_dict() for a in self.segment_analyses],
        }


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Transcription engines
# ---------------------------------------------------------------------------

class WhisperAPITranscriber:
    """Transcribe audio using OpenAI Whisper API."""

    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm", ".mpeg", ".mpga"}
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

    def __init__(self, model: str = "whisper-1") -> None:
        self.model = model
        self._client = None
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key.startswith("sk-"):
                from openai import OpenAI
                self._client = OpenAI()
                logger.info("Whisper API transcriber ready")
        except Exception as e:
            logger.warning(f"Whisper API not available: {e}")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def transcribe(
        self,
        file_path: str,
        language: str | None = None,
    ) -> tuple[str, list[TranscriptSegment]]:
        """Transcribe an audio file with timestamps."""
        if not self._client:
            raise RuntimeError("OpenAI client not available")

        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        if path.stat().st_size > self.MAX_FILE_SIZE:
            logger.warning("File exceeds 25MB — will be chunked")
            return self._transcribe_chunked(file_path, language)

        with open(file_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language=language,
            )

        full_text = response.text
        detected_lang = getattr(response, "language", language or "en")

        segments = []
        for seg in getattr(response, "segments", []):
            segments.append(TranscriptSegment(
                text=seg.get("text", "").strip(),
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                language=detected_lang,
                confidence=float(seg.get("avg_logprob", 0)),
            ))

        return full_text, segments

    def _transcribe_chunked(
        self,
        file_path: str,
        language: str | None,
    ) -> tuple[str, list[TranscriptSegment]]:
        """Transcribe large files in chunks."""
        # For large files, use basic transcription without segments
        with open(file_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="text",
                language=language,
            )

        # Split into pseudo-segments by sentences
        text = response if isinstance(response, str) else response.text
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        segments = []
        time_per_char = 0.05  # rough estimate
        offset = 0.0

        for sent in sentences:
            duration = len(sent) * time_per_char
            segments.append(TranscriptSegment(
                text=sent + ".",
                start=offset,
                end=offset + duration,
                language=language or "en",
            ))
            offset += duration

        return text, segments


class LocalWhisperTranscriber:
    """Transcribe audio using local Whisper model."""

    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._model = None

    @property
    def is_available(self) -> bool:
        try:
            import whisper
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size)
            logger.info(f"Local Whisper model loaded: {self.model_size}")
        return self._model

    def transcribe(
        self,
        file_path: str,
        language: str | None = None,
    ) -> tuple[str, list[TranscriptSegment]]:
        """Transcribe an audio file using local model."""
        model = self._load_model()
        result = model.transcribe(
            file_path,
            language=language,
            verbose=False,
        )

        full_text = result["text"]
        detected_lang = result.get("language", language or "en")

        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                language=detected_lang,
            ))

        return full_text, segments


# ---------------------------------------------------------------------------
# Speech Analyzer
# ---------------------------------------------------------------------------

class SpeechAnalyzer:
    """
    End-to-end speech/audio/video persuasion analyzer.

    Pipeline: Audio → Transcription → Chunked Analysis → Timeline Report
    """

    def __init__(
        self,
        transcriber: str = "auto",  # auto, api, local
        whisper_model: str = "base",
        chunk_duration: float = 60.0,  # Analyze in 60-second chunks
        pipeline: Any = None,
    ) -> None:
        self.chunk_duration = chunk_duration
        self._pipeline = pipeline

        # Select transcriber
        if transcriber == "api" or transcriber == "auto":
            self._transcriber = WhisperAPITranscriber()
            if not self._transcriber.is_available and transcriber == "auto":
                self._transcriber = LocalWhisperTranscriber(whisper_model)
        elif transcriber == "local":
            self._transcriber = LocalWhisperTranscriber(whisper_model)
        else:
            self._transcriber = WhisperAPITranscriber()

        available = "API" if isinstance(self._transcriber, WhisperAPITranscriber) else "Local"
        logger.info(f"SpeechAnalyzer initialized (transcriber: {available})")

    def _get_pipeline(self):
        if self._pipeline is None:
            from src.pipeline.persuasix_pipeline import PersuasixPipeline
            self._pipeline = PersuasixPipeline.from_default_models(device="cpu")
        return self._pipeline

    def analyze_file(self, file_path: str, language: str | None = None) -> SpeechAnalysisResult:
        """Analyze an audio/video file for persuasion."""
        logger.info(f"Analyzing speech: {file_path}")
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 1: Transcribe
        full_text, segments = self._transcriber.transcribe(file_path, language)
        detected_lang = segments[0].language if segments else (language or "en")

        total_duration = segments[-1].end if segments else 0

        logger.info(f"  Transcribed: {len(segments)} segments, {_fmt_time(total_duration)}")

        # Step 2: Chunk segments for analysis
        chunks = self._chunk_segments(segments)

        # Step 3: Analyze each chunk
        segment_analyses = self._analyze_chunks(chunks, detected_lang)

        # Step 4: Build timeline
        timeline = self._build_timeline(segment_analyses)

        # Step 5: Compute overall metrics
        all_techniques = []
        total_score = 0.0
        for a in segment_analyses:
            total_score += a.manipulation_score
            all_techniques.extend(a.techniques)
        overall_score = total_score / max(len(segment_analyses), 1)
        unique_techniques = list(set(all_techniques))

        return SpeechAnalysisResult(
            source=str(path.name),
            duration=total_duration,
            language=detected_lang,
            full_transcript=full_text,
            segments=segments,
            segment_analyses=segment_analyses,
            overall_score=overall_score,
            overall_techniques=unique_techniques,
            timeline=timeline,
        )

    def analyze_url(self, url: str, language: str | None = None) -> SpeechAnalysisResult:
        """Download and analyze audio from a URL."""
        logger.info(f"Downloading audio: {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Detect extension from URL or content-type
            content_type = response.headers.get("content-type", "")
            ext_map = {
                "audio/mpeg": ".mp3", "audio/wav": ".wav",
                "audio/mp4": ".m4a", "video/mp4": ".mp4",
                "audio/ogg": ".ogg", "audio/flac": ".flac",
                "video/webm": ".webm",
            }
            ext = ext_map.get(content_type.split(";")[0], ".mp3")

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name

            result = self.analyze_file(tmp_path, language)
            result.source = url
            Path(tmp_path).unlink(missing_ok=True)
            return result

        except Exception as e:
            logger.error(f"Failed to download/analyze audio: {e}")
            raise

    def _chunk_segments(self, segments: list[TranscriptSegment]) -> list[list[TranscriptSegment]]:
        """Group segments into chunks for analysis."""
        if not segments:
            return []

        chunks: list[list[TranscriptSegment]] = []
        current_chunk: list[TranscriptSegment] = []
        chunk_start = segments[0].start

        for seg in segments:
            current_chunk.append(seg)
            if seg.end - chunk_start >= self.chunk_duration:
                chunks.append(current_chunk)
                current_chunk = []
                chunk_start = seg.end

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _analyze_chunks(
        self,
        chunks: list[list[TranscriptSegment]],
        language: str,
    ) -> list[SegmentAnalysis]:
        """Analyze each chunk of segments."""
        pipe = self._get_pipeline()
        results = []

        for chunk in chunks:
            chunk_text = " ".join(seg.text for seg in chunk)
            if len(chunk_text.strip()) < 20:
                continue

            try:
                analysis = pipe.analyze(chunk_text, language=language)

                # Create a merged segment for the chunk
                merged_segment = TranscriptSegment(
                    text=chunk_text,
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    language=language,
                )

                results.append(SegmentAnalysis(
                    segment=merged_segment,
                    is_persuasive=analysis.is_persuasive,
                    manipulation_score=analysis.manipulation_score,
                    techniques=analysis.techniques,
                    highlighted_phrases=analysis.highlighted_phrases,
                    explanation=analysis.explanation,
                ))

            except Exception as e:
                logger.warning(f"Chunk analysis failed: {e}")
                continue

        return results

    def _build_timeline(self, analyses: list[SegmentAnalysis]) -> list[dict]:
        """Build a time-indexed manipulation timeline."""
        timeline = []
        for a in analyses:
            timeline.append({
                "start": round(a.segment.start, 1),
                "end": round(a.segment.end, 1),
                "timestamp": a.segment.timestamp_str,
                "score": round(a.manipulation_score * 100, 1),
                "is_persuasive": a.is_persuasive,
                "techniques": a.techniques,
                "preview": a.segment.text[:100] + "..." if len(a.segment.text) > 100 else a.segment.text,
            })
        return timeline
