"""PersuasiX Studio — Full Platform with URL Analysis, File Upload, Comparison, and more."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import gradio as gr

from components import (
    build_highlighted_text_html,
    build_technique_html,
    build_explanation_html,
    build_severity_gauge,
    build_comparison_html,
    build_article_meta_html,
    build_multi_comparison_html,
    build_batch_table_html,
    build_dashboard_html,
    build_intelligence_html,
    build_education_html,
    EXAMPLE_TEXTS,
    CSS,
)
from scraper import scrape_url
from file_parser import parse_file
from history import save_analysis, get_stats, clear_history, load_history


def create_demo() -> gr.Blocks:
    pipeline = None

    def get_pipeline():
        nonlocal pipeline
        if pipeline is None:
            from src.pipeline.persuasix_pipeline import PersuasixPipeline
            pipeline = PersuasixPipeline.from_default_models(device="cpu")
        return pipeline

    # ==================================================================
    # TAB 1: Text Analysis
    # ==================================================================
    def analyze_text(text: str, language: str, threshold: float):
        if not text or not text.strip():
            empty = (
                "<div class='glass-card' style='text-align:center;padding:60px;'>"
                "<p style='font-size:1.2em;color:#94a3b8;'>Paste text above and click <b>Analyze</b></p>"
                "</div>"
            )
            return empty, empty, empty, empty, empty, "{}", ""

        lang_map = {"English": "en", "Francais": "fr", "Arabic": "ar", "Spanish": "es", "German": "de", "Chinese": "zh", "Hindi": "hi"}
        lang = lang_map.get(language, "en")

        pipe = get_pipeline()
        pipe.detection_threshold = threshold
        result = pipe.analyze(text, language=lang)
        result_dict = result.to_dict()

        # Save to history
        save_analysis(result_dict, source_type="text", source_label=text[:80])

        highlighted_html = build_highlighted_text_html(
            result.text, result.highlighted_phrases, result.is_persuasive,
            result.manipulation_score, result.severity_score, result.techniques,
        )
        techniques_html = build_technique_html(result.techniques, result.technique_probabilities)
        explanation_html = build_explanation_html(
            result.explanation, result.cross_lingual_explanations, result.highlighted_phrases,
        )
        severity_html = build_severity_gauge(result.severity_score, result.manipulation_score)
        comparison_html = build_comparison_html(result.text, result.neutral_rewrite, result.manipulation_score)
        raw_json = json.dumps(result_dict, indent=2, ensure_ascii=False)

        n_tech = len(result.techniques)
        verdict = "Manipulative" if result.is_persuasive else "Neutral"
        status = f"Analysis complete — {verdict} | {n_tech} technique(s) | Severity: {result.severity_score}/5 | Score: {int(result.manipulation_score * 100)}/100"

        return highlighted_html, techniques_html, explanation_html, severity_html, comparison_html, raw_json, status

    # ==================================================================
    # TAB 2: URL Analyzer
    # ==================================================================
    def analyze_url(url: str, language: str):
        if not url or not url.strip():
            empty = "<div class='glass-card' style='text-align:center;padding:40px;'><p style='color:#94a3b8;'>Enter a URL above and click <b>Analyze URL</b></p></div>"
            return "", empty, empty, empty, ""

        article = scrape_url(url)
        if not article.success:
            error_html = f"""
            <div class="glass-card" style="text-align:center; padding:40px;">
                <div style="font-size:2em; margin-bottom:12px;">&#9888;</div>
                <h3 style="color:#f43f5e;">Scraping Failed</h3>
                <p style="color:#94a3b8;">{article.error}</p>
            </div>
            """
            return "", error_html, "", "", ""

        lang_map = {"English": "en", "Francais": "fr", "Arabic": "ar", "Spanish": "es", "German": "de", "Chinese": "zh", "Hindi": "hi"}
        lang = lang_map.get(language, article.language)

        pipe = get_pipeline()
        # Limit text to ~3000 words for API
        text = article.text
        words = text.split()
        if len(words) > 3000:
            text = " ".join(words[:3000])

        result = pipe.analyze(text, language=lang)
        result_dict = result.to_dict()
        save_analysis(result_dict, source_type="url", source_label=article.title or article.source)

        meta_html = build_article_meta_html(
            article.title, article.source, article.author,
            article.date, article.word_count, article.language,
        )
        highlighted_html = build_highlighted_text_html(
            result.text, result.highlighted_phrases, result.is_persuasive,
            result.manipulation_score, result.severity_score, result.techniques,
        )
        explanation_html = build_explanation_html(
            result.explanation, result.cross_lingual_explanations, result.highlighted_phrases,
        )
        comparison_html = build_comparison_html(result.text, result.neutral_rewrite, result.manipulation_score)

        verdict = "Manipulative" if result.is_persuasive else "Neutral"
        status = f"{article.source} — {verdict} | {len(result.techniques)} technique(s) | Score: {int(result.manipulation_score * 100)}/100"

        return meta_html, highlighted_html, explanation_html, comparison_html, status

    # ==================================================================
    # TAB 3: File Upload
    # ==================================================================
    def analyze_file(file, language: str):
        if file is None:
            empty = "<div class='glass-card' style='text-align:center;padding:40px;'><p style='color:#94a3b8;'>Upload a file (PDF, DOCX, or TXT) to analyze</p></div>"
            return "", empty, empty, empty, ""

        doc = parse_file(file.name if hasattr(file, 'name') else str(file))
        if not doc.success:
            error_html = f"""
            <div class="glass-card" style="text-align:center; padding:40px;">
                <div style="font-size:2em; margin-bottom:12px;">&#9888;</div>
                <h3 style="color:#f43f5e;">File Parsing Failed</h3>
                <p style="color:#94a3b8;">{doc.error}</p>
            </div>
            """
            return "", error_html, "", "", ""

        lang_map = {"English": "en", "Francais": "fr", "Arabic": "ar", "Spanish": "es", "German": "de", "Chinese": "zh", "Hindi": "hi"}
        lang = lang_map.get(language, "en")

        # Limit text
        text = doc.text
        words = text.split()
        if len(words) > 3000:
            text = " ".join(words[:3000])

        pipe = get_pipeline()
        result = pipe.analyze(text, language=lang)
        result_dict = result.to_dict()
        save_analysis(result_dict, source_type="file", source_label=doc.filename)

        file_info = f"""
        <div class="glass-card" style="margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:1.8em;">&#128196;</span>
                <div>
                    <h4 style="color:#f1f5f9; margin:0;">{doc.filename}</h4>
                    <span style="color:#94a3b8; font-size:0.9em;">
                        {doc.file_type.upper()} | {doc.word_count:,} words
                        {f' | {doc.page_count} pages' if doc.page_count else ''}
                    </span>
                </div>
            </div>
        </div>
        """
        highlighted_html = build_highlighted_text_html(
            result.text, result.highlighted_phrases, result.is_persuasive,
            result.manipulation_score, result.severity_score, result.techniques,
        )
        explanation_html = build_explanation_html(
            result.explanation, result.cross_lingual_explanations, result.highlighted_phrases,
        )
        comparison_html = build_comparison_html(result.text, result.neutral_rewrite, result.manipulation_score)

        verdict = "Manipulative" if result.is_persuasive else "Neutral"
        status = f"{doc.filename} — {verdict} | {len(result.techniques)} technique(s) | Score: {int(result.manipulation_score * 100)}/100"

        return file_info, highlighted_html, explanation_html, comparison_html, status

    # ==================================================================
    # TAB 4: Multi-Source Comparison
    # ==================================================================
    def compare_sources(source1: str, source2: str, source3: str, language: str, mode: str):
        sources = [s.strip() for s in [source1, source2, source3] if s and s.strip()]
        if len(sources) < 2:
            return "<div class='glass-card'><p style='color:#94a3b8;'>Enter at least 2 sources (URLs or texts) to compare.</p></div>", ""

        lang_map = {"English": "en", "Francais": "fr", "Arabic": "ar", "Spanish": "es", "German": "de", "Chinese": "zh", "Hindi": "hi"}
        lang = lang_map.get(language, "en")
        pipe = get_pipeline()

        results = []
        for src in sources:
            if mode == "URLs" and (src.startswith("http") or "." in src.split()[0]):
                article = scrape_url(src)
                if article.success:
                    text = article.text
                    words = text.split()
                    if len(words) > 2000:
                        text = " ".join(words[:2000])
                    r = pipe.analyze(text, language=lang)
                    d = r.to_dict()
                    d["source_label"] = article.title or article.source
                    results.append(d)
                else:
                    results.append({
                        "source_label": src[:40],
                        "is_persuasive": False, "techniques": [],
                        "severity_score": 0, "manipulation_score": 0,
                    })
            else:
                text = src
                words = text.split()
                if len(words) > 2000:
                    text = " ".join(words[:2000])
                r = pipe.analyze(text, language=lang)
                d = r.to_dict()
                d["source_label"] = src[:40]
                results.append(d)

        for d in results:
            save_analysis(d, source_type="comparison", source_label=d.get("source_label", ""))

        comparison_html = build_multi_comparison_html(results)
        status = f"Compared {len(results)} sources"
        return comparison_html, status

    # ==================================================================
    # TAB 5: Batch Analysis
    # ==================================================================
    def batch_analyze(texts_input: str, language: str):
        if not texts_input or not texts_input.strip():
            return "<div class='glass-card'><p style='color:#94a3b8;'>Enter texts separated by blank lines.</p></div>", ""

        # Split by double newline
        texts = [t.strip() for t in texts_input.split("\n\n") if t.strip()]
        if not texts:
            texts = [t.strip() for t in texts_input.split("\n") if t.strip() and len(t.strip()) > 20]

        if not texts:
            return "<div class='glass-card'><p style='color:#94a3b8;'>No valid texts found. Separate texts with blank lines.</p></div>", ""

        lang_map = {"English": "en", "Francais": "fr", "Arabic": "ar", "Spanish": "es", "German": "de", "Chinese": "zh", "Hindi": "hi"}
        lang = lang_map.get(language, "en")
        pipe = get_pipeline()

        results = []
        for text in texts[:20]:  # Max 20 texts
            r = pipe.analyze(text, language=lang)
            d = r.to_dict()
            results.append(d)
            save_analysis(d, source_type="batch", source_label=text[:60])

        table_html = build_batch_table_html(results)
        status = f"Batch analysis complete — {len(results)} texts analyzed"
        return table_html, status

    # ==================================================================
    # TAB 6: Dashboard
    # ==================================================================
    def load_dashboard():
        stats = get_stats()
        return build_dashboard_html(stats)

    def clear_dashboard():
        clear_history()
        return build_dashboard_html(get_stats())

    # ==================================================================
    # TAB 7: Intelligence
    # ==================================================================
    def load_intelligence():
        from src.pipeline.intelligence import PersuasixIntelligence

        history = load_history()
        for index, entry in enumerate(history, start=1):
            entry.setdefault("id", index)
        report = PersuasixIntelligence(history).threat_report()
        return build_intelligence_html(report)

    # ==================================================================
    # BUILD UI
    # ==================================================================
    with gr.Blocks(title="PersuasiX Studio") as demo:

        # ── Hero header ──
        gr.HTML("""
        <div class="hero-section">
            <h1>PersuasiX</h1>
            <p class="subtitle">Multilingual Persuasion Detection, Explanation &amp; Neutralization Platform</p>
            <div class="tech-badges">
                <span>GPT-4o-mini</span>
                <span>RoBERTa</span>
                <span>FLAN-T5</span>
                <span>URL Scraping</span>
                <span>PDF/DOCX</span>
                <span>Multi-Source</span>
                <span>18 Techniques</span>
                <span>7 Languages</span>
                <span>Analyst Intel</span>
            </div>
            <div class="stats-bar">
                <div class="stat-item"><div class="stat-value">18</div><div class="stat-label">Techniques</div></div>
                <div class="stat-item"><div class="stat-value">7</div><div class="stat-label">Languages</div></div>
                <div class="stat-item"><div class="stat-value">10</div><div class="stat-label">Modes</div></div>
                <div class="stat-item"><div class="stat-value">5</div><div class="stat-label">AI Systems</div></div>
            </div>
        </div>
        """)

        # ── Main Tabs ──
        with gr.Tabs():

            # ============================================================
            # TAB 1: Text Analysis
            # ============================================================
            with gr.TabItem("Text Analysis", id=0):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Paste a news article, tweet, political speech, or any text...",
                            lines=6, max_lines=12,
                        )
                        with gr.Row():
                            lang_text = gr.Dropdown(choices=["English", "Francais", "Arabic", "Spanish", "German", "Chinese", "Hindi"], value="English", label="Language", scale=1)
                            threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Sensitivity", scale=2)
                        with gr.Row():
                            analyze_btn = gr.Button("Analyze Text", variant="primary", size="lg", scale=2)
                            clear_btn = gr.Button("Clear", size="lg", scale=1)
                    with gr.Column(scale=1, min_width=220):
                        gr.HTML('<h4 style="color:#a5b4fc; margin:0 0 12px; font-size:0.85em; text-transform:uppercase; letter-spacing:1.5px;">Quick Examples</h4>')
                        for label, text in EXAMPLE_TEXTS.items():
                            gr.Button(label, size="sm", elem_classes=["example-btn"]).click(fn=lambda t=text: t, outputs=text_input)

                status_text = gr.Textbox(label="Status", interactive=False, max_lines=1)
                with gr.Tabs():
                    with gr.TabItem("Highlighted Analysis"):
                        text_highlighted = gr.HTML()
                    with gr.TabItem("Techniques"):
                        text_techniques = gr.HTML()
                    with gr.TabItem("Explanation"):
                        text_explanation = gr.HTML()
                    with gr.TabItem("Severity"):
                        text_severity = gr.HTML()
                    with gr.TabItem("Original vs Neutral"):
                        text_comparison = gr.HTML()
                    with gr.TabItem("JSON"):
                        text_json = gr.Code(language="json")

                text_outputs = [text_highlighted, text_techniques, text_explanation, text_severity, text_comparison, text_json, status_text]
                analyze_btn.click(fn=analyze_text, inputs=[text_input, lang_text, threshold], outputs=text_outputs)
                clear_btn.click(fn=lambda: ("", "", "", "", "", "", "", ""), outputs=[text_input] + text_outputs)

            # ============================================================
            # TAB 2: URL Analyzer
            # ============================================================
            with gr.TabItem("URL Analyzer", id=1):
                gr.HTML("""
                <div class="glass-card" style="margin-bottom:16px;">
                    <h3 style="color:#f1f5f9; margin:0 0 6px;">&#127760; Analyze Any Web Article</h3>
                    <p style="color:#94a3b8; margin:0; font-size:0.9em;">Paste a news article URL — we'll scrape, extract, and analyze it for manipulation techniques.</p>
                </div>
                """)
                with gr.Row():
                    url_input = gr.Textbox(label="Article URL", placeholder="https://www.example.com/article...", scale=3)
                    lang_url = gr.Dropdown(choices=["English", "Francais", "Arabic", "Spanish", "German", "Chinese", "Hindi"], value="English", label="Language", scale=1)
                url_btn = gr.Button("Analyze URL", variant="primary", size="lg")
                status_url = gr.Textbox(label="Status", interactive=False, max_lines=1)

                url_meta = gr.HTML()
                with gr.Tabs():
                    with gr.TabItem("Highlighted Analysis"):
                        url_highlighted = gr.HTML()
                    with gr.TabItem("Explanation"):
                        url_explanation = gr.HTML()
                    with gr.TabItem("Neutral Rewrite"):
                        url_comparison = gr.HTML()

                url_btn.click(fn=analyze_url, inputs=[url_input, lang_url], outputs=[url_meta, url_highlighted, url_explanation, url_comparison, status_url])

            # ============================================================
            # TAB 3: File Upload
            # ============================================================
            with gr.TabItem("File Upload", id=2):
                gr.HTML("""
                <div class="glass-card" style="margin-bottom:16px;">
                    <h3 style="color:#f1f5f9; margin:0 0 6px;">&#128196; Analyze Documents</h3>
                    <p style="color:#94a3b8; margin:0; font-size:0.9em;">Upload a PDF, DOCX, or TXT file for persuasion analysis.</p>
                </div>
                """)
                with gr.Row():
                    file_input = gr.File(label="Upload Document", file_types=[".pdf", ".docx", ".txt", ".html", ".md"], scale=2)
                    lang_file = gr.Dropdown(choices=["English", "Francais", "Arabic", "Spanish", "German", "Chinese", "Hindi"], value="English", label="Language", scale=1)
                file_btn = gr.Button("Analyze Document", variant="primary", size="lg")
                status_file = gr.Textbox(label="Status", interactive=False, max_lines=1)

                file_info = gr.HTML()
                with gr.Tabs():
                    with gr.TabItem("Highlighted Analysis"):
                        file_highlighted = gr.HTML()
                    with gr.TabItem("Explanation"):
                        file_explanation = gr.HTML()
                    with gr.TabItem("Neutral Rewrite"):
                        file_comparison = gr.HTML()

                file_btn.click(fn=analyze_file, inputs=[file_input, lang_file], outputs=[file_info, file_highlighted, file_explanation, file_comparison, status_file])

            # ============================================================
            # TAB 4: Compare Sources
            # ============================================================
            with gr.TabItem("Compare Sources", id=3):
                gr.HTML("""
                <div class="glass-card" style="margin-bottom:16px;">
                    <h3 style="color:#f1f5f9; margin:0 0 6px;">&#9878; Multi-Source Comparison</h3>
                    <p style="color:#94a3b8; margin:0; font-size:0.9em;">Compare 2-3 sources (URLs or texts) side-by-side to see which is more manipulative.</p>
                </div>
                """)
                with gr.Row():
                    compare_mode = gr.Radio(choices=["URLs", "Texts"], value="Texts", label="Input Mode", scale=1)
                    lang_compare = gr.Dropdown(choices=["English", "Francais", "Arabic", "Spanish", "German", "Chinese", "Hindi"], value="English", label="Language", scale=1)
                source1 = gr.Textbox(label="Source 1", placeholder="URL or text...", lines=3)
                source2 = gr.Textbox(label="Source 2", placeholder="URL or text...", lines=3)
                source3 = gr.Textbox(label="Source 3 (optional)", placeholder="URL or text...", lines=3)
                compare_btn = gr.Button("Compare Sources", variant="primary", size="lg")
                status_compare = gr.Textbox(label="Status", interactive=False, max_lines=1)
                compare_output = gr.HTML()

                compare_btn.click(fn=compare_sources, inputs=[source1, source2, source3, lang_compare, compare_mode], outputs=[compare_output, status_compare])

            # ============================================================
            # TAB 5: Batch Analysis
            # ============================================================
            with gr.TabItem("Batch Analysis", id=4):
                gr.HTML("""
                <div class="glass-card" style="margin-bottom:16px;">
                    <h3 style="color:#f1f5f9; margin:0 0 6px;">&#128218; Batch Analysis</h3>
                    <p style="color:#94a3b8; margin:0; font-size:0.9em;">Analyze multiple texts at once. Separate each text with a blank line. Max 20 texts.</p>
                </div>
                """)
                batch_input = gr.Textbox(label="Texts (separated by blank lines)", lines=10, max_lines=20, placeholder="First text here...\n\nSecond text here...\n\nThird text here...")
                with gr.Row():
                    lang_batch = gr.Dropdown(choices=["English", "Francais", "Arabic", "Spanish", "German", "Chinese", "Hindi"], value="English", label="Language", scale=1)
                    batch_btn = gr.Button("Analyze All", variant="primary", size="lg", scale=2)
                status_batch = gr.Textbox(label="Status", interactive=False, max_lines=1)
                batch_output = gr.HTML()

                batch_btn.click(fn=batch_analyze, inputs=[batch_input, lang_batch], outputs=[batch_output, status_batch])

            # ============================================================
            # TAB 6: Dashboard
            # ============================================================
            with gr.TabItem("Dashboard", id=5):
                gr.HTML("""
                <div class="glass-card" style="margin-bottom:16px;">
                    <h3 style="color:#f1f5f9; margin:0 0 6px;">&#128202; Analytics Dashboard</h3>
                    <p style="color:#94a3b8; margin:0; font-size:0.9em;">Track your analysis history, see trends, and aggregate statistics.</p>
                </div>
                """)
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Dashboard", size="lg", scale=2)
                    clear_hist_btn = gr.Button("Clear History", size="lg", scale=1)
                dashboard_output = gr.HTML(value=build_dashboard_html(get_stats()))

                refresh_btn.click(fn=load_dashboard, outputs=[dashboard_output])
                clear_hist_btn.click(fn=clear_dashboard, outputs=[dashboard_output])

            # ============================================================
            # TAB 7: Intelligence
            # ============================================================
            with gr.TabItem("Intelligence", id=6):
                gr.HTML("""
                <div class="section-header">
                    <div>
                        <h3>&#128300; Analyst Intelligence</h3>
                        <p>Review active-learning candidates, drift, repeated narratives, and high-risk signals.</p>
                    </div>
                    <span class="section-pill">History Powered</span>
                </div>
                """)
                with gr.Row():
                    refresh_intel_btn = gr.Button("Refresh Intelligence", variant="primary", size="lg")
                intelligence_output = gr.HTML(value=load_intelligence())
                refresh_intel_btn.click(fn=load_intelligence, outputs=[intelligence_output])

            # ============================================================
            # TAB 8: Fact Checker
            # ============================================================
            with gr.TabItem("Fact Checker", id=7):
                gr.HTML("""
                <div class="glass-card" style="margin-bottom:16px;">
                    <h3 style="color:#f1f5f9; margin:0 0 6px;">&#128269; Fact Checker</h3>
                    <p style="color:#94a3b8; margin:0; font-size:0.9em;">Extract claims from text and verify them against fact-checking databases and AI analysis.</p>
                </div>
                """)
                fact_input = gr.Textbox(label="Text to Fact-Check", lines=5, placeholder="Paste an article or statement with factual claims...")
                with gr.Row():
                    lang_fact = gr.Dropdown(choices=["English", "Francais", "Arabic", "Spanish", "German", "Chinese", "Hindi"], value="English", label="Language", scale=1)
                    fact_btn = gr.Button("Check Facts", variant="primary", size="lg", scale=2)
                status_fact = gr.Textbox(label="Status", interactive=False, max_lines=1)
                fact_output = gr.HTML()

                def run_fact_check(text, language):
                    if not text or not text.strip():
                        return "<div class='glass-card'><p style='color:#94a3b8;'>Enter text above to fact-check.</p></div>", ""
                    lang_map = {"English": "en", "Francais": "fr", "Arabic": "ar", "Spanish": "es", "German": "de", "Chinese": "zh", "Hindi": "hi"}
                    lang = lang_map.get(language, "en")
                    pipe = get_pipeline()
                    report = pipe.fact_check(text, language=lang)
                    if "error" in report:
                        return f"<div class='glass-card'><p style='color:#f43f5e;'>Error: {report['error']}</p></div>", "Error"
                    # Build HTML
                    score = report.get("credibility_score", 50)
                    if score >= 70: color = "#22c55e"
                    elif score >= 40: color = "#f59e0b"
                    else: color = "#f43f5e"
                    html = f"""
                    <div class='glass-card' style='text-align:center;padding:24px;'>
                        <div style='font-size:48px;font-weight:800;color:{color};'>{score:.0f}/100</div>
                        <div style='color:{color};font-size:16px;font-weight:600;margin-top:4px;'>Credibility Score</div>
                        <div style='margin-top:8px;color:#94a3b8;font-size:13px;'>
                            {report.get('num_claims',0)} claims found &bull;
                            {report.get('num_verified',0)} verified &bull;
                            {report.get('num_false',0)} false &bull;
                            {report.get('num_unverified',0)} unverified
                        </div>
                    </div>
                    """
                    for r in report.get("results", []):
                        v = r.get("verdict","unverified")
                        vc = {"true":"#22c55e","mostly_true":"#86efac","mixed":"#f59e0b","mostly_false":"#fb923c","false":"#ef4444"}.get(v,"#94a3b8")
                        html += f"""
                        <div class='glass-card' style='margin-top:10px;border-left:3px solid {vc};'>
                            <div style='font-size:11px;color:{vc};font-weight:700;text-transform:uppercase;letter-spacing:1px;'>{v.replace('_',' ')}</div>
                            <div style='color:#f1f5f9;margin:6px 0;font-style:italic;'>"{r.get('claim','')[:150]}"</div>
                            <div style='color:#94a3b8;font-size:12px;'>{r.get('explanation','')}</div>
                            {"<div style='margin-top:4px;'><a href='"+r.get('source_url','')+"' target='_blank' style='color:#818cf8;font-size:11px;'>Source: "+r.get('publisher','')+"</a></div>" if r.get('source_url') else ""}
                        </div>
                        """
                    status = f"Fact check complete — Credibility: {score:.0f}/100 | {report.get('num_claims',0)} claims analyzed"
                    return html, status

                fact_btn.click(fn=run_fact_check, inputs=[fact_input, lang_fact], outputs=[fact_output, status_fact])

            # ============================================================
            # TAB 9: Span Detection
            # ============================================================
            with gr.TabItem("Span Detection", id=8):
                gr.HTML("""
                <div class="glass-card" style="margin-bottom:16px;">
                    <h3 style="color:#f1f5f9; margin:0 0 6px;">&#127919; Span-Level Detection</h3>
                    <p style="color:#94a3b8; margin:0; font-size:0.9em;">Identify the EXACT words and phrases that are manipulative, with character-level precision.</p>
                </div>
                """)
                span_input = gr.Textbox(label="Text to Analyze", lines=5, placeholder="Paste text to identify exact manipulative spans...")
                span_btn = gr.Button("Detect Spans", variant="primary", size="lg")
                status_span = gr.Textbox(label="Status", interactive=False, max_lines=1)
                span_output = gr.HTML()

                def run_span_detection(text):
                    if not text or not text.strip():
                        return "<div class='glass-card'><p style='color:#94a3b8;'>Enter text above to detect spans.</p></div>", ""
                    pipe = get_pipeline()
                    result = pipe.detect_spans(text)
                    if "error" in result:
                        return f"<div class='glass-card'><p style='color:#f43f5e;'>Error: {result['error']}</p></div>", "Error"
                    spans = result.get("spans", [])
                    coverage = result.get("coverage", 0)
                    technique_colors = {
                        "appeal_to_fear": "#ef4444", "loaded_language": "#f97316",
                        "appeal_to_emotion": "#ec4899", "bandwagon": "#8b5cf6",
                        "false_dilemma": "#06b6d4", "ad_hominem": "#f43f5e",
                        "appeal_to_authority": "#a855f7", "exaggeration": "#eab308",
                        "doubt": "#64748b", "whataboutism": "#14b8a6",
                        "name_calling": "#e11d48", "flag_waving": "#3b82f6",
                    }
                    # Build annotated text
                    annotated = text
                    for s in sorted(spans, key=lambda x: -x.get("start", 0)):
                        start = s.get("start", 0)
                        end = s.get("end", 0)
                        tech = s.get("technique", "")
                        color = technique_colors.get(tech, "#818cf8")
                        evidence = s.get("evidence", "").replace('"', '&quot;')
                        span_text = annotated[start:end]
                        replacement = f'<span style="background:{color}33;border-bottom:2px solid {color};padding:1px 3px;border-radius:3px;cursor:help;" title="[{tech.replace("_"," ")}] {evidence}">{span_text}</span>'
                        annotated = annotated[:start] + replacement + annotated[end:]
                    html = f"""
                    <div class='glass-card'>
                        <div style='margin-bottom:12px;'>
                            <span style='color:#818cf8;font-weight:600;'>{len(spans)} spans</span>
                            <span style='color:#94a3b8;'> detected | </span>
                            <span style='color:#818cf8;font-weight:600;'>{coverage*100:.1f}%</span>
                            <span style='color:#94a3b8;'> text coverage | </span>
                            <span style='color:#818cf8;font-weight:600;'>{len(result.get("techniques_found",[]))}</span>
                            <span style='color:#94a3b8;'> techniques</span>
                        </div>
                        <div style='line-height:1.8;color:#e2e8f0;font-size:14px;'>{annotated}</div>
                    </div>
                    """
                    # Span details
                    for s in spans:
                        tech = s.get("technique", "")
                        color = technique_colors.get(tech, "#818cf8")
                        html += f"""
                        <div class='glass-card' style='margin-top:8px;border-left:3px solid {color};padding:10px 14px;'>
                            <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <span style='color:{color};font-weight:700;font-size:12px;text-transform:uppercase;'>{tech.replace('_',' ')}</span>
                                <span style='color:#94a3b8;font-size:11px;'>chars {s.get("start",0)}-{s.get("end",0)} | {s.get("confidence",0)*100:.0f}%</span>
                            </div>
                            <div style='color:#f1f5f9;margin:4px 0;font-style:italic;'>"{s.get("text","")}"</div>
                            <div style='color:#94a3b8;font-size:12px;'>{s.get("evidence","")}</div>
                        </div>
                        """
                    status = f"Span detection complete — {len(spans)} spans found | {coverage*100:.1f}% coverage"
                    return html, status

                span_btn.click(fn=run_span_detection, inputs=[span_input], outputs=[span_output, status_span])

            # ============================================================
            # TAB 10: Education Center
            # ============================================================
            with gr.TabItem("Learn", id=9):
                gr.HTML(build_education_html())

        # ── Footer ──
        gr.HTML("""
        <div class="footer">
            <p>PersuasiX Studio &mdash; Full Platform for Persuasion Analysis</p>
            <p style="margin-top:4px;">Text &bull; URL &bull; File &bull; Compare &bull; Batch &bull; Dashboard &bull; Intelligence &bull; Fact Check &bull; Span Detection &bull; Education</p>
            <p style="margin-top:4px; font-size:0.8em;">Powered by GPT-4o-mini, RoBERTa, FLAN-T5, Sentence-Transformers &amp; Gradio</p>
        </div>
        """)

    return demo


def main():
    demo = create_demo()
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7863"))
    demo.launch(share=False, server_port=port, css=CSS)


if __name__ == "__main__":
    main()
