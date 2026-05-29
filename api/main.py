"""PersuasiX REST API — FastAPI application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.database import init_db
from api.routes.analysis import router as analysis_router
from api.routes.intelligence import router as intelligence_router
from api.routes.monitor import router as monitor_router
from api.routes.reports import router as reports_router
from api.routes.social import router as social_router
from api.routes.speech import router as speech_router
from api.schemas import HealthResponse
from app.annotation import create_annotation_routes

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PersuasiX API",
    description=(
        "REST API for multilingual persuasion detection, explanation, "
        "and neutralization. Powered by GPT-4o-mini, RoBERTa, and FLAN-T5."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(analysis_router)
app.include_router(intelligence_router)
app.include_router(monitor_router)
app.include_router(reports_router)
app.include_router(create_annotation_routes())
app.include_router(social_router)
app.include_router(speech_router)

# Static file serving for reports
reports_dir = Path(__file__).resolve().parent.parent / "data" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")


# ---------------------------------------------------------------------------
# Health & Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
def root():
    return {
        "name": "PersuasiX API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "analyze_text": "POST /api/v1/analyze/text",
            "analyze_url": "POST /api/v1/analyze/url",
            "analyze_file": "POST /api/v1/analyze/file",
            "analyze_batch": "POST /api/v1/analyze/batch",
            "history": "GET /api/v1/history",
            "stats": "GET /api/v1/stats",
            "monitor_feeds": "GET /api/v1/monitor/feeds",
            "alerts": "GET /api/v1/monitor/alerts",
            "annotation_tasks": "GET /api/v1/annotate/tasks",
            "social_status": "GET /api/v1/social/status",
            "social_search": "POST /api/v1/social/search",
            "speech_file": "POST /api/v1/speech/file",
            "speech_url": "POST /api/v1/speech/url",
            "generate_report": "POST /api/v1/reports/generate",
            "fact_check": "POST /api/v1/fact-check",
            "span_detection": "POST /api/v1/detect-spans",
            "active_learning": "GET /api/v1/intelligence/active-learning",
            "drift_report": "GET /api/v1/intelligence/drift",
            "narrative_clusters": "GET /api/v1/intelligence/narrative-clusters",
            "threat_report": "GET /api/v1/intelligence/threat-report",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "status": "healthy",
        "version": "2.0.0",
        "pipeline_ready": True,
        "database": "sqlite",
        "openai_available": bool(openai_key and openai_key.startswith("sk-")),
    }


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    init_db()
    # Start RSS monitor in background
    try:
        from monitor.rss_monitor import start_monitor_scheduler
        start_monitor_scheduler()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
