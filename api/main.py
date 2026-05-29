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
from api.routes.monitor import router as monitor_router
from api.routes.reports import router as reports_router
from api.schemas import HealthResponse

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
app.include_router(monitor_router)
app.include_router(reports_router)

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
            "generate_report": "POST /api/v1/reports/generate",
            "fact_check": "POST /api/v1/fact-check",
            "span_detection": "POST /api/v1/detect-spans",
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
