"""PDF report generator for PersuasiX analysis results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger


REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TECHNIQUE_NAMES = {
    "appeal_to_fear": "Appeal to Fear",
    "appeal_to_authority": "Appeal to Authority",
    "bandwagon": "Bandwagon",
    "false_dilemma": "False Dilemma",
    "ad_hominem": "Ad Hominem",
    "straw_man": "Straw Man",
    "red_herring": "Red Herring",
    "loaded_language": "Loaded Language",
    "whataboutism": "Whataboutism",
    "causal_oversimplification": "Causal Oversimplification",
    "appeal_to_emotion": "Appeal to Emotion",
    "repetition": "Repetition",
    "exaggeration": "Exaggeration",
    "doubt": "Doubt",
    "slogans": "Slogans",
    "name_calling": "Name Calling",
    "flag_waving": "Flag Waving",
    "thought_terminating_cliche": "Thought-Terminating Cliche",
}


def generate_pdf_report(analysis: dict) -> tuple[Path, str]:
    """Generate a professional PDF report from an analysis result."""
    from fpdf import FPDF

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_id = analysis.get("id", "unknown")
    filename = f"persuasix_report_{analysis_id}_{ts}.pdf"
    filepath = REPORTS_DIR / filename

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # -- Header --
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 45, "F")
    pdf.set_text_color(129, 140, 248)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_y(10)
    pdf.cell(0, 12, "PersuasiX", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, "Persuasion Analysis Report", ln=True, align="C")
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)

    # -- Summary Box --
    pdf.set_text_color(0, 0, 0)
    is_manip = analysis.get("is_persuasive", False)
    score = int(analysis.get("manipulation_score", 0) * 100)
    severity = analysis.get("severity_score", 0)
    techniques = analysis.get("techniques", [])

    verdict = "MANIPULATIVE" if is_manip else "NEUTRAL"
    if severity <= 1:
        level = "Low"
    elif severity <= 2.5:
        level = "Moderate"
    elif severity <= 4:
        level = "High"
    else:
        level = "Critical"

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Analysis Summary", ln=True)
    pdf.set_draw_color(129, 140, 248)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(55, 7, "Verdict:")
    pdf.set_font("Helvetica", "B", 11)
    if is_manip:
        pdf.set_text_color(220, 38, 38)
    else:
        pdf.set_text_color(34, 197, 94)
    pdf.cell(0, 7, verdict, ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(55, 7, "Manipulation Score:")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, f"{score}/100", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(55, 7, "Severity:")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, f"{severity:.1f}/5 ({level})", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(55, 7, "Techniques Found:")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, str(len(techniques)), ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(55, 7, "Language:")
    lang_map = {"en": "English", "fr": "French", "ar": "Arabic"}
    pdf.cell(0, 7, lang_map.get(analysis.get("language", "en"), "English"), ln=True)
    pdf.ln(6)

    # -- Techniques --
    if techniques:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Detected Techniques", ln=True)
        pdf.set_draw_color(129, 140, 248)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        probs = analysis.get("technique_probabilities", {})
        for tech in techniques:
            label = TECHNIQUE_NAMES.get(tech, tech.replace("_", " ").title())
            prob = probs.get(tech, 0)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(99, 102, 241)
            pdf.cell(80, 6, f"  {label}", ln=False)
            pdf.set_text_color(100, 100, 100)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Confidence: {int(prob*100)}%", ln=True)
        pdf.ln(4)

    # -- Highlighted Phrases --
    phrases = analysis.get("highlighted_phrases", [])
    if phrases:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Evidence & Manipulative Phrases", ln=True)
        pdf.set_draw_color(129, 140, 248)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        for h in phrases:
            tech = TECHNIQUE_NAMES.get(h.get("technique", ""), h.get("technique", ""))
            phrase = h.get("phrase", "")
            evidence = h.get("evidence", "")

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(220, 38, 38)
            pdf.cell(0, 6, f"[{tech}]", ln=True)

            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.set_x(15)
            pdf.multi_cell(180, 5, f'"{phrase}"')

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.set_x(15)
            pdf.multi_cell(180, 5, f"Why: {evidence}")
            pdf.ln(2)
        pdf.ln(2)

    # -- Explanation --
    explanation = analysis.get("explanation", "")
    if explanation:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Explanation", ln=True)
        pdf.set_draw_color(129, 140, 248)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, explanation)
        pdf.ln(4)

    # -- Neutral Rewrite --
    neutral = analysis.get("neutral_rewrite", "")
    if neutral and neutral != analysis.get("text", ""):
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Neutral Rewrite", ln=True)
        pdf.set_draw_color(34, 197, 94)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, neutral)
        pdf.ln(4)

    # -- Original Text --
    text = analysis.get("text", "")
    if text:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Original Text", ln=True)
        pdf.set_draw_color(129, 140, 248)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        # Limit to avoid huge PDFs
        display_text = text[:3000]
        if len(text) > 3000:
            display_text += "\n\n[... text truncated for report ...]"
        pdf.multi_cell(0, 4.5, display_text)

    # -- Footer on each page --
    for page_num in range(1, pdf.pages_count + 1):
        pdf.page = page_num
        pdf.set_y(-15)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 10, f"PersuasiX Report | Page {page_num}/{pdf.pages_count}", align="C")

    pdf.output(str(filepath))
    logger.info(f"PDF report generated: {filepath}")
    return filepath, filename
