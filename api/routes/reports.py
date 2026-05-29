"""PDF report generation routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.database import Analysis, Report, get_db
from api.schemas import ReportRequest, ReportResponse

router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])


@router.post("/generate", response_model=ReportResponse)
def generate_report(req: ReportRequest, db: Session = Depends(get_db)):
    """Generate a PDF report for an analysis."""
    from reports.pdf_generator import generate_pdf_report

    analysis = db.query(Analysis).filter(Analysis.id == req.analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    filepath, filename = generate_pdf_report(analysis.to_dict())

    record = Report(
        analysis_id=req.analysis_id,
        filename=filename,
        filepath=str(filepath),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    resp = record.to_dict()
    resp["download_url"] = f"/api/v1/reports/{record.id}/download"
    return resp


@router.get("/{report_id}/download")
def download_report(report_id: int, db: Session = Depends(get_db)):
    """Download a generated PDF report."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    path = Path(report.filepath)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file not found on disk")

    return FileResponse(
        path=str(path),
        filename=report.filename,
        media_type="application/pdf",
    )


@router.get("/", response_model=list[ReportResponse])
def list_reports(db: Session = Depends(get_db)):
    """List all generated reports."""
    reports = db.query(Report).order_by(Report.created_at.desc()).limit(50).all()
    result = []
    for r in reports:
        d = r.to_dict()
        d["download_url"] = f"/api/v1/reports/{r.id}/download"
        result.append(d)
    return result
