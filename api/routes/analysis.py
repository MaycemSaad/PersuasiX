"""Analysis API routes — text, URL, file, batch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from api.database import Analysis, get_db, save_analysis_to_db
from api.schemas import (
    AnalysisListResponse,
    AnalysisResponse,
    BatchAnalysisRequest,
    StatsResponse,
    TextAnalysisRequest,
    URLAnalysisRequest,
)

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

# Pipeline singleton
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline.persuasix_pipeline import PersuasixPipeline
        _pipeline = PersuasixPipeline.from_default_models(device="cpu")
    return _pipeline


@router.post("/analyze/text", response_model=AnalysisResponse)
def analyze_text(req: TextAnalysisRequest):
    """Analyze a text for persuasion techniques."""
    pipe = _get_pipeline()
    pipe.detection_threshold = req.threshold
    result = pipe.analyze(req.text, language=req.language)
    record = save_analysis_to_db(result.to_dict(), source_type="text", source_label=req.text[:80])
    return record.to_dict()


@router.post("/analyze/url", response_model=AnalysisResponse)
def analyze_url(req: URLAnalysisRequest):
    """Scrape and analyze an article from a URL."""
    from app.scraper import scrape_url

    article = scrape_url(req.url)
    if not article.success:
        raise HTTPException(status_code=422, detail=f"Failed to scrape URL: {article.error}")

    text = article.text
    words = text.split()
    if len(words) > 3000:
        text = " ".join(words[:3000])

    lang = req.language or article.language
    pipe = _get_pipeline()
    result = pipe.analyze(text, language=lang)
    record = save_analysis_to_db(
        result.to_dict(),
        source_type="url",
        source_label=article.title or article.source,
    )
    resp = record.to_dict()
    resp["article_meta"] = {
        "title": article.title,
        "source": article.source,
        "author": article.author,
        "date": article.date,
        "word_count": article.word_count,
    }
    return resp


@router.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    """Upload and analyze a document (PDF, DOCX, TXT)."""
    import tempfile
    from app.file_parser import parse_file

    suffix = Path(file.filename or "file.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    doc = parse_file(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if not doc.success:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {doc.error}")

    text = doc.text
    words = text.split()
    if len(words) > 3000:
        text = " ".join(words[:3000])

    pipe = _get_pipeline()
    result = pipe.analyze(text, language=language)
    record = save_analysis_to_db(result.to_dict(), source_type="file", source_label=doc.filename)
    return record.to_dict()


@router.post("/analyze/batch")
def analyze_batch(req: BatchAnalysisRequest):
    """Analyze multiple texts at once."""
    pipe = _get_pipeline()
    results = []
    for text in req.texts[:20]:
        result = pipe.analyze(text, language=req.language)
        record = save_analysis_to_db(result.to_dict(), source_type="batch", source_label=text[:60])
        results.append(record.to_dict())
    return {"total": len(results), "items": results}


# ---------------------------------------------------------------------------
# History & Stats
# ---------------------------------------------------------------------------

@router.get("/history", response_model=AnalysisListResponse)
def list_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    source_type: str | None = None,
    db: Session = Depends(get_db),
):
    """List analysis history with pagination."""
    query = db.query(Analysis).order_by(Analysis.created_at.desc())
    if source_type:
        query = query.filter(Analysis.source_type == source_type)
    total = query.count()
    items = query.offset(skip).limit(limit).all()
    return {"total": total, "items": [a.to_dict() for a in items]}


@router.get("/history/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Get a specific analysis by ID."""
    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record.to_dict()


@router.delete("/history/{analysis_id}")
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Delete an analysis from history."""
    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    db.delete(record)
    db.commit()
    return {"detail": "Analysis deleted"}


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get aggregate statistics."""
    from api.database import get_db_stats
    return get_db_stats()


# ---------------------------------------------------------------------------
# Fact Checking & Span Detection
# ---------------------------------------------------------------------------

@router.post("/fact-check")
def fact_check(req: TextAnalysisRequest):
    """Extract and verify factual claims in a text."""
    pipe = _get_pipeline()
    report = pipe.fact_check(req.text, language=req.language)
    return report


@router.post("/detect-spans")
def detect_spans(req: TextAnalysisRequest):
    """Detect exact manipulative spans in text with character offsets."""
    pipe = _get_pipeline()
    result = pipe.detect_spans(req.text)
    return result
