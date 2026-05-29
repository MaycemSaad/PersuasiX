"""Speech-to-text persuasion analysis routes."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from src.pipeline.speech_analyzer import SpeechAnalyzer

router = APIRouter(prefix="/api/v1/speech", tags=["Speech Analysis"])

_analyzer: SpeechAnalyzer | None = None


def _get_analyzer(transcriber: str = "auto") -> SpeechAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SpeechAnalyzer(transcriber=transcriber)
    return _analyzer


class SpeechURLRequest(BaseModel):
    url: str = Field(..., min_length=8)
    language: str | None = None
    transcriber: str = Field("auto", pattern="^(auto|api|local)$")


@router.post("/file")
async def analyze_speech_file(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    transcriber: str = Form("auto"),
):
    """Upload an audio/video file, transcribe it, and analyze persuasive segments."""
    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = _get_analyzer(transcriber).analyze_file(tmp_path, language)
        return result.to_dict()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/url")
def analyze_speech_url(req: SpeechURLRequest):
    """Download an audio/video URL, transcribe it, and analyze persuasive segments."""
    result = _get_analyzer(req.transcriber).analyze_url(req.url, req.language)
    return result.to_dict()
