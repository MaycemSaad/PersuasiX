"""Pydantic schemas for PersuasiX REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to analyze")
    language: str = Field("en", description="Language code: en, fr, ar")
    threshold: float = Field(0.5, ge=0.1, le=0.9, description="Detection sensitivity")

class URLAnalysisRequest(BaseModel):
    url: str = Field(..., description="Article URL to scrape and analyze")
    language: str = Field("en", description="Language code: en, fr, ar")

class BatchAnalysisRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=20)
    language: str = Field("en")

class HighlightedPhrase(BaseModel):
    phrase: str
    technique: str
    evidence: str

class AnalysisResponse(BaseModel):
    id: int
    text: str
    language: str
    is_persuasive: bool
    techniques: list[str]
    technique_probabilities: dict[str, float]
    highlighted_phrases: list[HighlightedPhrase]
    explanation: str
    neutral_rewrite: str
    severity_score: float
    manipulation_score: float
    cross_lingual_explanations: dict[str, str]
    source_type: str
    source_label: str
    created_at: Optional[str] = None

class AnalysisListResponse(BaseModel):
    total: int
    items: list[AnalysisResponse]

class StatsResponse(BaseModel):
    total_analyses: int
    manipulative_count: int
    neutral_count: int
    avg_severity: float
    avg_manipulation: float
    technique_frequency: dict[str, int]
    source_breakdown: dict[str, int]
    unread_alerts: int


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class FeedCreateRequest(BaseModel):
    url: str = Field(..., description="RSS/Atom feed URL")
    name: str = Field("", description="Friendly name for the feed")
    language: str = Field("en")
    check_interval_min: int = Field(30, ge=5, le=1440)

class FeedResponse(BaseModel):
    id: int
    url: str
    name: str
    language: str
    check_interval_min: int
    last_checked: Optional[str] = None
    active: bool
    created_at: Optional[str] = None

class AlertResponse(BaseModel):
    id: int
    feed_id: int
    article_url: str
    article_title: str
    manipulation_score: float
    severity_score: float
    techniques: list[str]
    analysis_id: Optional[int] = None
    read: bool
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    analysis_id: int = Field(..., description="ID of the analysis to generate report for")

class ReportResponse(BaseModel):
    id: int
    analysis_id: Optional[int] = None
    filename: str
    download_url: str
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    pipeline_ready: bool
    database: str
    openai_available: bool

class ErrorResponse(BaseModel):
    detail: str
