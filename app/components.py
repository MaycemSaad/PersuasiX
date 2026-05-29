"""UI components and styling for PersuasiX Studio."""

from __future__ import annotations

import html as _html

CSS = """
/* ── Base ── */
.gradio-container {
    max-width: 1280px !important;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
}
body { background: #0f172a !important; }

/* ── Hero ── */
.hero-section {
    text-align: center;
    padding: 40px 20px 30px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}
.hero-section h1 {
    font-size: 3em;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero-section .subtitle {
    color: #94a3b8;
    font-size: 1.1em;
    margin: 8px 0 20px;
}
.tech-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.tech-badges span {
    padding: 5px 16px;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    color: #a5b4fc;
    font-size: 0.85em;
    font-weight: 600;
}
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 40px;
}
.stat-item { text-align: center; }
.stat-value {
    font-size: 1.8em;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    color: #64748b;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin: 8px 0;
}

/* ── Highlight tooltips ── */
.hl-phrase {
    position: relative;
    cursor: help;
    padding: 2px 4px;
    border-radius: 4px;
    transition: filter 0.2s;
}
.hl-phrase:hover { filter: brightness(1.3); }
.hl-phrase .hl-tip {
    display: none;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 10px 14px;
    min-width: 260px;
    max-width: 380px;
    z-index: 100;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    color: #e2e8f0;
    font-size: 0.85em;
    line-height: 1.5;
    text-align: left;
    pointer-events: none;
}
.hl-phrase:hover .hl-tip { display: block; }

/* ── Example buttons ── */
.example-btn {
    width: 100%;
    text-align: left !important;
    background: rgba(30,41,59,0.6) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    margin: 4px 0 !important;
    transition: all 0.2s !important;
}
.example-btn:hover {
    background: rgba(99,102,241,0.2) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #a5b4fc !important;
}

/* ── Technique cards ── */
.tech-card {
    background: rgba(15,23,42,0.6);
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
    border-left: 4px solid;
    transition: transform 0.2s, box-shadow 0.2s;
}
.tech-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 24px;
    color: #475569;
    font-size: 0.85em;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 30px;
}

/* Interface polish overrides */
.gradio-container {
    background: #101735 !important;
}
body {
    background: #101735 !important;
}
.hero-section {
    background: linear-gradient(135deg, rgba(40,54,113,0.92), rgba(50,38,97,0.92));
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 8px;
    box-shadow: 0 18px 42px rgba(0,0,0,0.24);
    padding: 34px 20px 28px;
}
.hero-section h1 {
    letter-spacing: 0;
}
.tech-badges span,
.glass-card,
.example-btn,
.tech-card,
.gradio-container button {
    border-radius: 8px !important;
}
button[role="tab"] {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}
button[role="tab"][aria-selected="true"] {
    background: rgba(249,115,22,0.12) !important;
    border-color: #f97316 !important;
    color: #fed7aa !important;
}
.tab-nav button {
    color: #cbd5e1 !important;
}
.stats-bar {
    flex-wrap: wrap;
}
.glass-card {
    background: rgba(24, 31, 53, 0.82);
    border: 1px solid rgba(148,163,184,0.14);
    box-shadow: 0 10px 28px rgba(0,0,0,0.18);
}
.section-header {
    align-items: center;
    background: rgba(24,31,53,0.72);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 8px;
    display: flex;
    gap: 16px;
    justify-content: space-between;
    margin-bottom: 16px;
    padding: 18px 20px;
}
.section-header h3 {
    color: #f1f5f9;
    margin: 0 0 4px;
}
.section-header p {
    color: #94a3b8;
    font-size: 0.9em;
    margin: 0;
}
.section-pill {
    background: rgba(99,102,241,0.16);
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 8px;
    color: #a5b4fc;
    font-size: 0.82em;
    font-weight: 700;
    padding: 7px 12px;
    white-space: nowrap;
}
.intel-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
}
.intel-card {
    background: rgba(15,23,42,0.54);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 8px;
    padding: 16px;
}
.intel-kpi {
    color: #f8fafc;
    font-size: 1.8em;
    font-weight: 800;
}
.intel-label {
    color: #94a3b8;
    font-size: 0.78em;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
@media (max-width: 900px) {
    .hero-section { padding: 26px 14px 22px; }
    .hero-section h1 { font-size: 2.25em; }
    .stats-bar { gap: 18px; }
    .section-header { align-items: flex-start; flex-direction: column; }
    .intel-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 560px) {
    .intel-grid { grid-template-columns: 1fr; }
    .glass-card { padding: 16px; }
}
"""

TECHNIQUE_COLORS: dict[str, str] = {
    "appeal_to_fear": "#ef4444",
    "appeal_to_authority": "#3b82f6",
    "bandwagon": "#a855f7",
    "false_dilemma": "#f97316",
    "ad_hominem": "#dc2626",
    "straw_man": "#ea580c",
    "red_herring": "#7c3aed",
    "loaded_language": "#2563eb",
    "whataboutism": "#14b8a6",
    "causal_oversimplification": "#eab308",
    "appeal_to_emotion": "#ec4899",
    "repetition": "#78716c",
    "exaggeration": "#f43f5e",
    "doubt": "#64748b",
    "slogans": "#fb923c",
    "name_calling": "#e11d48",
    "flag_waving": "#22c55e",
    "thought_terminating_cliche": "#94a3b8",
}

TECHNIQUE_ICONS: dict[str, str] = {
    "appeal_to_fear": "&#9888;",
    "appeal_to_authority": "&#128081;",
    "bandwagon": "&#128101;",
    "false_dilemma": "&#8644;",
    "ad_hominem": "&#128074;",
    "straw_man": "&#127806;",
    "red_herring": "&#128031;",
    "loaded_language": "&#128163;",
    "whataboutism": "&#128073;",
    "causal_oversimplification": "&#128279;",
    "appeal_to_emotion": "&#128148;",
    "repetition": "&#128257;",
    "exaggeration": "&#128200;",
    "doubt": "&#10067;",
    "slogans": "&#128227;",
    "name_calling": "&#128680;",
    "flag_waving": "&#127988;",
    "thought_terminating_cliche": "&#128683;",
}

TECHNIQUE_DESCRIPTIONS: dict[str, str] = {
    "appeal_to_fear": "Instills fear to influence the audience's decision",
    "appeal_to_authority": "Uses authority figures to lend credibility without evidence",
    "bandwagon": "Suggests everyone agrees, pressuring conformity",
    "false_dilemma": "Presents only two options when more exist",
    "ad_hominem": "Attacks the person rather than the argument",
    "straw_man": "Misrepresents someone's argument to make it easier to attack",
    "red_herring": "Introduces an irrelevant topic to divert attention",
    "loaded_language": "Uses emotionally charged words to manipulate",
    "whataboutism": "Deflects criticism by pointing to others' faults",
    "causal_oversimplification": "Reduces complex issues to a single cause",
    "appeal_to_emotion": "Exploits emotions instead of using logic",
    "repetition": "Repeats a message to make it seem more true",
    "exaggeration": "Overstates facts to create a stronger impression",
    "doubt": "Questions credibility without evidence",
    "slogans": "Uses catchy phrases to replace critical thinking",
    "name_calling": "Labels opponents with negative terms",
    "flag_waving": "Exploits patriotism to justify a position",
    "thought_terminating_cliche": "Uses cliches to shut down debate",
}

EXAMPLE_TEXTS: dict[str, str] = {
    "Fear + Authority (EN)": (
        "If we don't act NOW, our children will inherit a wasteland. "
        "Every scientist agrees — this is our last chance."
    ),
    "False Dilemma (EN)": (
        "You're either with us or against us. There is no middle ground "
        "when the future of our nation is at stake."
    ),
    "Loaded Language (FR)": (
        "Ne soyez pas naifs ! Ce politicien est un menteur notoire qui ne cherche "
        "qu'a remplir ses poches. Reveillez-vous !"
    ),
    "Propaganda (AR)": (
        "هذا القرار سيدمر مستقبل أطفالنا! كل الخبراء يؤكدون ذلك. "
        "من يعارض هذا الرأي هو عدو للشعب."
    ),
    "Neutral text (EN)": (
        "The committee voted 7-5 in favor of the amendment after "
        "three hours of debate."
    ),
}


def _esc(text: str) -> str:
    return _html.escape(str(text))


def _esc_attr(text: str) -> str:
    return _html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# 1. Highlighted Text (primary output)
# ---------------------------------------------------------------------------

def build_highlighted_text_html(
    text: str,
    highlighted_phrases: list[dict],
    is_persuasive: bool,
    manipulation_score: float,
    severity: float,
    techniques: list[str],
) -> str:
    """Render original text with manipulative phrases highlighted + decision banner."""

    if not is_persuasive:
        return """
        <div class="glass-card" style="text-align:center; padding:50px;">
            <div style="font-size:3em; margin-bottom:16px;">&#9989;</div>
            <h2 style="color:#4ade80; margin:0 0 8px;">No Manipulation Detected</h2>
            <p style="color:#94a3b8; font-size:1.05em;">This text appears to be neutral and objective.</p>
            <div style="margin-top:20px; padding:12px 24px; display:inline-block; background:rgba(74,222,128,0.1); border:1px solid rgba(74,222,128,0.3); border-radius:12px;">
                <span style="color:#4ade80; font-weight:700;">Manipulation Score: 0/100</span>
            </div>
        </div>
        """

    manip_pct = int(manipulation_score * 100)
    if severity <= 1:
        level_color, level_label = "#facc15", "LOW"
    elif severity <= 2.5:
        level_color, level_label = "#fb923c", "MODERATE"
    elif severity <= 4:
        level_color, level_label = "#f43f5e", "HIGH"
    else:
        level_color, level_label = "#dc2626", "CRITICAL"

    # -- Decision banner --
    banner = f"""
    <div class="glass-card" style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px; margin-bottom:16px;">
        <div>
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.6em;">&#128680;</span>
                <span style="font-size:1.3em; font-weight:700; color:{level_color};">MANIPULATIVE TEXT DETECTED</span>
            </div>
            <p style="color:#94a3b8; margin:6px 0 0; font-size:0.95em;">
                {len(techniques)} technique(s) identified &mdash; Severity: {level_label}
            </p>
        </div>
        <div style="text-align:center;">
            <div style="position:relative; width:90px; height:90px;">
                <svg viewBox="0 0 36 36" style="width:90px; height:90px; transform:rotate(-90deg);">
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                          fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="3"/>
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                          fill="none" stroke="{level_color}" stroke-width="3"
                          stroke-dasharray="{manip_pct}, 100" stroke-linecap="round"/>
                </svg>
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:1.3em; font-weight:800; color:{level_color};">
                    {manip_pct}
                </div>
            </div>
            <div style="color:#94a3b8; font-size:0.75em; margin-top:4px;">MANIPULATION<br>SCORE</div>
        </div>
    </div>
    """

    # -- Build highlighted text --
    if not highlighted_phrases:
        body = f'<p style="color:#e2e8f0; line-height:1.8;">{_esc(text)}</p>'
    else:
        annotations = []
        for h in highlighted_phrases:
            phrase = h.get("phrase", "")
            start = text.find(phrase)
            if start >= 0:
                annotations.append({
                    "start": start,
                    "end": start + len(phrase),
                    "phrase": phrase,
                    "technique": h.get("technique", ""),
                    "evidence": h.get("evidence", ""),
                })

        annotations.sort(key=lambda a: (a["start"], -(a["end"] - a["start"])))
        filtered = []
        last_end = 0
        for ann in annotations:
            if ann["start"] >= last_end:
                filtered.append(ann)
                last_end = ann["end"]

        parts: list[str] = []
        pos = 0
        for ann in filtered:
            if ann["start"] > pos:
                parts.append(_esc(text[pos : ann["start"]]))

            color = TECHNIQUE_COLORS.get(ann["technique"], "#ef4444")
            label = ann["technique"].replace("_", " ").title()
            icon = TECHNIQUE_ICONS.get(ann["technique"], "")
            evidence_esc = _esc_attr(ann["evidence"])

            parts.append(
                f'<span class="hl-phrase" style="background:{color}22; border-bottom:2px solid {color};">'
                f'{_esc(ann["phrase"])}'
                f'<span class="hl-tip">'
                f'<strong style="color:{color};">{icon} {_esc(label)}</strong><br>'
                f'{_esc(ann["evidence"])}'
                f'</span>'
                f'</span>'
            )
            pos = ann["end"]

        if pos < len(text):
            parts.append(_esc(text[pos:]))

        body = "".join(parts)

    # -- Legend --
    used_techs = list(dict.fromkeys(h.get("technique", "") for h in highlighted_phrases))
    legend_items = ""
    for tech in used_techs:
        if not tech:
            continue
        color = TECHNIQUE_COLORS.get(tech, "#ef4444")
        icon = TECHNIQUE_ICONS.get(tech, "")
        label = tech.replace("_", " ").title()
        legend_items += (
            f'<span style="display:inline-flex; align-items:center; gap:5px; margin:4px 10px 4px 0; '
            f'padding:5px 12px; background:{color}15; border:1px solid {color}40; border-radius:8px; '
            f'font-size:0.85em; color:{color};">'
            f'{icon} {_esc(label)}'
            f'</span>'
        )

    text_block = f"""
    <div class="glass-card">
        <h3 style="color:#f1f5f9; margin:0 0 16px; font-size:1.1em;">
            &#128221; Annotated Text &mdash; Hover highlighted phrases for details
        </h3>
        <div style="background:rgba(15,23,42,0.6); padding:20px; border-radius:12px; line-height:2; font-size:1.05em; color:#e2e8f0;">
            {body}
        </div>
        <div style="margin-top:16px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.08);">
            <span style="color:#64748b; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Techniques found:</span><br>
            {legend_items}
        </div>
    </div>
    """

    return banner + text_block


# ---------------------------------------------------------------------------
# 2. Techniques Detected
# ---------------------------------------------------------------------------

def build_technique_html(techniques: list[str], probabilities: dict[str, float]) -> str:
    if not techniques:
        return """
        <div class="glass-card" style="text-align:center; padding:40px;">
            <div style="font-size:2.5em; margin-bottom:12px;">&#9989;</div>
            <h2 style="color:#4ade80; margin:0 0 8px;">No Persuasion Techniques Detected</h2>
            <p style="color:#94a3b8;">This text appears to be neutral and objective.</p>
        </div>
        """

    html = '<div style="display:flex; flex-direction:column; gap:10px;">'
    for tech in sorted(techniques, key=lambda t: -probabilities.get(t, 0)):
        color = TECHNIQUE_COLORS.get(tech, "#666")
        prob = probabilities.get(tech, 0)
        label = tech.replace("_", " ").title()
        icon = TECHNIQUE_ICONS.get(tech, "")
        desc = TECHNIQUE_DESCRIPTIONS.get(tech, "")
        pct = int(prob * 100)

        html += f"""
        <div class="tech-card" style="border-left-color:{color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:1.3em;">{icon}</span>
                    <span style="font-weight:700; color:#f1f5f9; font-size:1.05em;">{_esc(label)}</span>
                </div>
                <span style="font-weight:800; color:{color}; font-size:1.15em;">{pct}%</span>
            </div>
            <div style="margin:10px 0 6px;">
                <div style="background:rgba(255,255,255,0.06); border-radius:6px; height:8px; overflow:hidden;">
                    <div style="width:{pct}%; background:linear-gradient(90deg, {color}, {color}aa); height:100%; border-radius:6px; transition:width 0.5s;"></div>
                </div>
            </div>
            <p style="margin:0; color:#94a3b8; font-size:0.9em;">{_esc(desc)}</p>
        </div>
        """

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# 3. Explanation with evidence
# ---------------------------------------------------------------------------

def build_explanation_html(
    explanation: str,
    cross_lingual: dict[str, str],
    highlighted_phrases: list[dict] | None = None,
) -> str:
    if not explanation:
        return '<p style="color:#64748b;">No explanation generated (text is neutral).</p>'

    lang_meta = {
        "en": ("English", "ltr", "#3b82f6", "&#127468;&#127463;"),
        "fr": ("Francais", "ltr", "#f97316", "&#127467;&#127479;"),
        "ar": ("العربية", "rtl", "#22c55e", "&#127462;&#127466;"),
    }

    html = ""

    # -- Per-technique evidence --
    if highlighted_phrases:
        html += '<div class="glass-card" style="margin-bottom:16px;">'
        html += '<h3 style="color:#f1f5f9; margin:0 0 16px;">&#128269; Evidence &amp; Linguistic Cues</h3>'
        for h in highlighted_phrases:
            tech = h.get("technique", "")
            phrase = h.get("phrase", "")
            evidence = h.get("evidence", "")
            color = TECHNIQUE_COLORS.get(tech, "#ef4444")
            icon = TECHNIQUE_ICONS.get(tech, "")
            label = tech.replace("_", " ").title()

            html += f"""
            <div style="margin:10px 0; padding:14px; background:rgba(15,23,42,0.5);
                        border-radius:10px; border-left:4px solid {color};">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <span style="font-size:1.1em;">{icon}</span>
                    <strong style="color:{color};">{_esc(label)}</strong>
                </div>
                <div style="background:{color}15; padding:10px 14px; border-radius:8px; margin-bottom:8px;">
                    <span style="color:#94a3b8; font-size:0.8em; text-transform:uppercase;">Phrase:</span>
                    <p style="margin:4px 0 0; color:#f1f5f9; font-style:italic;">&ldquo;{_esc(phrase)}&rdquo;</p>
                </div>
                <p style="margin:0; color:#cbd5e1; font-size:0.95em;">
                    <strong style="color:#94a3b8;">Why manipulative:</strong> {_esc(evidence)}
                </p>
            </div>
            """
        html += "</div>"

    # -- Multilingual explanations --
    html += '<div class="glass-card">'
    html += '<h3 style="color:#f1f5f9; margin:0 0 16px;">&#127760; Multilingual Explanation</h3>'

    for lang in ["en", "fr", "ar"]:
        text = cross_lingual.get(lang, "")
        if not text:
            continue
        label, direction, color, flag = lang_meta.get(lang, (lang, "ltr", "#666", ""))

        html += f"""
        <div style="margin:10px 0; padding:16px; background:rgba(15,23,42,0.5);
                    border-radius:12px; border-left:4px solid {color};">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                <span style="font-size:1.2em;">{flag}</span>
                <strong style="color:{color}; font-size:1em;">{_esc(label)}</strong>
            </div>
            <p style="margin:0; direction:{direction}; color:#e2e8f0; line-height:1.7;">{_esc(text)}</p>
        </div>
        """

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# 4. Severity Gauge
# ---------------------------------------------------------------------------

def build_severity_gauge(severity: float, manipulation_score: float = 0.0) -> str:
    manip_pct = int(manipulation_score * 100)

    if severity <= 1:
        color, label = "#22c55e", "Low"
    elif severity <= 2.5:
        color, label = "#eab308", "Moderate"
    elif severity <= 4:
        color, label = "#f43f5e", "High"
    else:
        color, label = "#dc2626", "Critical"

    pct = int((severity / 5) * 100)

    return f"""
    <div class="glass-card" style="text-align:center; padding:40px;">
        <h3 style="color:#f1f5f9; margin:0 0 24px;">Manipulation Severity Assessment</h3>

        <div style="position:relative; width:180px; height:180px; margin:0 auto 24px;">
            <svg viewBox="0 0 36 36" style="width:180px; height:180px; transform:rotate(-90deg);">
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="2.5"/>
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none" stroke="{color}" stroke-width="2.5"
                      stroke-dasharray="{pct}, 100" stroke-linecap="round"/>
            </svg>
            <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);">
                <div style="font-size:2.8em; font-weight:800; color:{color};">{severity:.1f}</div>
                <div style="color:#64748b; font-size:0.85em;">/5</div>
            </div>
        </div>

        <div style="display:inline-block; padding:8px 24px; background:{color}20; border:1px solid {color}60; border-radius:10px;">
            <span style="font-weight:700; color:{color}; font-size:1.1em;">{label}</span>
        </div>

        <div style="display:flex; justify-content:center; gap:40px; margin-top:28px;">
            <div>
                <div style="color:#64748b; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Severity</div>
                <div style="font-size:1.4em; font-weight:700; color:#f1f5f9;">{severity:.1f}/5</div>
            </div>
            <div style="width:1px; background:rgba(255,255,255,0.08);"></div>
            <div>
                <div style="color:#64748b; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Manipulation</div>
                <div style="font-size:1.4em; font-weight:700; color:#f1f5f9;">{manip_pct}%</div>
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 5. Original vs Neutral comparison
# ---------------------------------------------------------------------------

def build_comparison_html(original: str, neutral: str, manipulation_score: float) -> str:
    manip_pct = int(manipulation_score * 100)

    return f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
        <div class="glass-card" style="border-left:4px solid #f43f5e;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:1.2em;">&#128308;</span>
                <h4 style="color:#f43f5e; margin:0;">Original (Persuasive)</h4>
            </div>
            <p style="color:#e2e8f0; line-height:1.7;">{_esc(original)}</p>
        </div>
        <div class="glass-card" style="border-left:4px solid #22c55e;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:1.2em;">&#128994;</span>
                <h4 style="color:#22c55e; margin:0;">Neutralized Version</h4>
            </div>
            <p style="color:#e2e8f0; line-height:1.7;">{_esc(neutral)}</p>
        </div>
    </div>
    <div style="text-align:center; margin-top:16px;">
        <div class="glass-card" style="display:inline-block; padding:12px 28px;">
            <span style="color:#94a3b8;">Manipulation distance:&nbsp;</span>
            <strong style="color:#f43f5e;">{manip_pct}%</strong>
            <span style="color:rgba(255,255,255,0.15); margin:0 12px;">|</span>
            <span style="color:#94a3b8;">Semantic preservation:&nbsp;</span>
            <strong style="color:#22c55e;">{100 - manip_pct}%</strong>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 6. URL Article Metadata
# ---------------------------------------------------------------------------

def build_article_meta_html(title: str, source: str, author: str, date: str, word_count: int, language: str) -> str:
    """Render article metadata card."""
    lang_labels = {"en": "English", "fr": "Francais", "ar": "Arabic"}
    return f"""
    <div class="glass-card" style="margin-bottom:16px;">
        <div style="display:flex; align-items:flex-start; gap:16px; flex-wrap:wrap;">
            <div style="flex:1; min-width:200px;">
                <h3 style="color:#f1f5f9; margin:0 0 8px; font-size:1.15em;">{_esc(title or 'Untitled Article')}</h3>
                <div style="display:flex; flex-wrap:wrap; gap:12px; color:#94a3b8; font-size:0.9em;">
                    <span>&#127760; {_esc(source)}</span>
                    {f'<span>&#128100; {_esc(author)}</span>' if author else ''}
                    {f'<span>&#128197; {_esc(date)}</span>' if date else ''}
                    <span>&#128196; {word_count:,} words</span>
                    <span>&#127463; {lang_labels.get(language, language)}</span>
                </div>
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 7. Multi-Source Comparison
# ---------------------------------------------------------------------------

def build_multi_comparison_html(results: list[dict]) -> str:
    """Build a side-by-side comparison of multiple analyzed sources."""
    if not results:
        return '<div class="glass-card"><p style="color:#94a3b8;">No results to compare.</p></div>'

    n = len(results)

    # -- Summary cards --
    cards_html = f'<div style="display:grid; grid-template-columns:repeat({min(n, 3)}, 1fr); gap:16px; margin-bottom:20px;">'
    for i, r in enumerate(results):
        score = int(r.get("manipulation_score", 0) * 100)
        severity = r.get("severity_score", 0)
        tech_count = len(r.get("techniques", []))
        label = r.get("source_label", f"Source {i+1}")
        is_manip = r.get("is_persuasive", False)

        if severity <= 1:
            color = "#22c55e"
        elif severity <= 2.5:
            color = "#eab308"
        elif severity <= 4:
            color = "#f43f5e"
        else:
            color = "#dc2626"

        verdict = "MANIPULATIVE" if is_manip else "NEUTRAL"
        verdict_color = color if is_manip else "#4ade80"

        cards_html += f"""
        <div class="glass-card" style="border-top:4px solid {color};">
            <h4 style="color:#f1f5f9; margin:0 0 12px; font-size:0.95em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{_esc_attr(label)}">{_esc(label[:40])}</h4>
            <div style="text-align:center; margin:12px 0;">
                <div style="position:relative; width:70px; height:70px; margin:0 auto;">
                    <svg viewBox="0 0 36 36" style="width:70px; height:70px; transform:rotate(-90deg);">
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="3"/>
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none" stroke="{color}" stroke-width="3"
                              stroke-dasharray="{score}, 100" stroke-linecap="round"/>
                    </svg>
                    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:1.1em; font-weight:800; color:{color};">{score}</div>
                </div>
            </div>
            <div style="text-align:center;">
                <span style="display:inline-block; padding:4px 12px; background:{verdict_color}20; border:1px solid {verdict_color}40; border-radius:6px; font-size:0.8em; font-weight:700; color:{verdict_color};">{verdict}</span>
            </div>
            <div style="margin-top:12px; font-size:0.85em; color:#94a3b8;">
                <div>Techniques: <strong style="color:#f1f5f9;">{tech_count}</strong></div>
                <div>Severity: <strong style="color:#f1f5f9;">{severity:.1f}/5</strong></div>
            </div>
        </div>
        """
    cards_html += "</div>"

    # -- Comparison bar chart --
    bar_html = '<div class="glass-card"><h4 style="color:#f1f5f9; margin:0 0 16px;">Manipulation Score Comparison</h4>'
    for i, r in enumerate(results):
        score = int(r.get("manipulation_score", 0) * 100)
        label = r.get("source_label", f"Source {i+1}")
        severity = r.get("severity_score", 0)
        if severity <= 1:
            color = "#22c55e"
        elif severity <= 2.5:
            color = "#eab308"
        elif severity <= 4:
            color = "#f43f5e"
        else:
            color = "#dc2626"

        bar_html += f"""
        <div style="margin:8px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#cbd5e1; font-size:0.9em;">{_esc(label[:35])}</span>
                <span style="color:{color}; font-weight:700;">{score}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.06); border-radius:6px; height:12px; overflow:hidden;">
                <div style="width:{score}%; background:linear-gradient(90deg, {color}, {color}99); height:100%; border-radius:6px;"></div>
            </div>
        </div>
        """
    bar_html += "</div>"

    # -- Technique overlap --
    all_techs: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        for t in r.get("techniques", []):
            if t not in all_techs:
                all_techs[t] = []
            all_techs[t].append(i)

    if all_techs:
        overlap_html = '<div class="glass-card" style="margin-top:16px;"><h4 style="color:#f1f5f9; margin:0 0 12px;">Technique Distribution Across Sources</h4>'
        for tech, sources in sorted(all_techs.items(), key=lambda x: -len(x[1])):
            color = TECHNIQUE_COLORS.get(tech, "#666")
            label = tech.replace("_", " ").title()
            dots = ""
            for i in range(n):
                if i in sources:
                    dots += f'<span style="width:18px; height:18px; border-radius:50%; background:{color}; display:inline-block; margin:0 3px; font-size:0.65em; line-height:18px; text-align:center; color:white;">{i+1}</span>'
                else:
                    dots += f'<span style="width:18px; height:18px; border-radius:50%; background:rgba(255,255,255,0.06); display:inline-block; margin:0 3px;"></span>'
            overlap_html += f"""
            <div style="display:flex; align-items:center; gap:12px; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:{color}; font-size:0.9em; min-width:180px;">{_esc(label)}</span>
                <div>{dots}</div>
            </div>
            """
        overlap_html += "</div>"
    else:
        overlap_html = ""

    # -- Verdict --
    if n >= 2:
        scores = [(r.get("manipulation_score", 0), r.get("source_label", f"Source {i+1}")) for i, r in enumerate(results)]
        scores.sort(key=lambda x: -x[0])
        most_manip = scores[0]
        least_manip = scores[-1]
        verdict_html = f"""
        <div class="glass-card" style="margin-top:16px; text-align:center;">
            <h4 style="color:#f1f5f9; margin:0 0 12px;">Verdict</h4>
            <p style="color:#cbd5e1; font-size:1.05em; line-height:1.7;">
                <strong style="color:#f43f5e;">{_esc(most_manip[1][:40])}</strong> is the most manipulative source
                (score: {int(most_manip[0]*100)}%), while
                <strong style="color:#22c55e;">{_esc(least_manip[1][:40])}</strong> is the least
                (score: {int(least_manip[0]*100)}%).
            </p>
        </div>
        """
    else:
        verdict_html = ""

    return cards_html + bar_html + overlap_html + verdict_html


# ---------------------------------------------------------------------------
# 8. Batch Analysis Table
# ---------------------------------------------------------------------------

def build_batch_table_html(results: list[dict]) -> str:
    """Build an HTML table for batch analysis results."""
    if not results:
        return '<div class="glass-card"><p style="color:#94a3b8;">No results.</p></div>'

    html = """
    <div class="glass-card" style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
            <thead>
                <tr style="border-bottom:2px solid rgba(255,255,255,0.1);">
                    <th style="padding:10px; text-align:left; color:#94a3b8;">#</th>
                    <th style="padding:10px; text-align:left; color:#94a3b8;">Text Preview</th>
                    <th style="padding:10px; text-align:center; color:#94a3b8;">Verdict</th>
                    <th style="padding:10px; text-align:center; color:#94a3b8;">Score</th>
                    <th style="padding:10px; text-align:center; color:#94a3b8;">Techniques</th>
                    <th style="padding:10px; text-align:center; color:#94a3b8;">Severity</th>
                </tr>
            </thead>
            <tbody>
    """

    for i, r in enumerate(results):
        text_preview = r.get("text", "")[:80] + ("..." if len(r.get("text", "")) > 80 else "")
        is_manip = r.get("is_persuasive", False)
        score = int(r.get("manipulation_score", 0) * 100)
        n_tech = len(r.get("techniques", []))
        severity = r.get("severity_score", 0)

        verdict_color = "#f43f5e" if is_manip else "#4ade80"
        verdict_text = "Manipulative" if is_manip else "Neutral"

        html += f"""
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:10px; color:#64748b;">{i+1}</td>
                <td style="padding:10px; color:#e2e8f0; max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{_esc(text_preview)}</td>
                <td style="padding:10px; text-align:center;"><span style="padding:3px 10px; border-radius:12px; background:{verdict_color}20; color:{verdict_color}; font-size:0.85em; font-weight:600;">{verdict_text}</span></td>
                <td style="padding:10px; text-align:center; font-weight:700; color:{'#f43f5e' if score > 50 else '#eab308' if score > 20 else '#4ade80'};">{score}%</td>
                <td style="padding:10px; text-align:center; color:#f1f5f9;">{n_tech}</td>
                <td style="padding:10px; text-align:center; color:#f1f5f9;">{severity:.1f}/5</td>
            </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    """

    # -- Summary stats --
    total = len(results)
    manip_count = sum(1 for r in results if r.get("is_persuasive"))
    avg_score = sum(r.get("manipulation_score", 0) for r in results) / total if total else 0

    html += f"""
    <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:16px;">
        <div class="glass-card" style="text-align:center; padding:16px;">
            <div style="font-size:2em; font-weight:800; color:#818cf8;">{total}</div>
            <div style="color:#64748b; font-size:0.85em;">Texts Analyzed</div>
        </div>
        <div class="glass-card" style="text-align:center; padding:16px;">
            <div style="font-size:2em; font-weight:800; color:#f43f5e;">{manip_count}</div>
            <div style="color:#64748b; font-size:0.85em;">Manipulative</div>
        </div>
        <div class="glass-card" style="text-align:center; padding:16px;">
            <div style="font-size:2em; font-weight:800; color:#eab308;">{int(avg_score*100)}%</div>
            <div style="color:#64748b; font-size:0.85em;">Avg Score</div>
        </div>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# 9. Dashboard Stats
# ---------------------------------------------------------------------------

def build_dashboard_html(stats: dict) -> str:
    """Render the analytics dashboard."""
    total = stats.get("total_analyses", 0)
    if total == 0:
        return """
        <div class="glass-card" style="text-align:center; padding:50px;">
            <div style="font-size:3em; margin-bottom:16px;">&#128202;</div>
            <h2 style="color:#f1f5f9; margin:0 0 8px;">No Analysis History Yet</h2>
            <p style="color:#94a3b8;">Start analyzing texts to build your dashboard.</p>
        </div>
        """

    manip = stats.get("manipulative_count", 0)
    neutral = stats.get("neutral_count", 0)
    avg_sev = stats.get("avg_severity", 0)
    avg_manip = stats.get("avg_manipulation", 0)

    # -- Top stats --
    html = f"""
    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:20px;">
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.2em; font-weight:800; color:#818cf8;">{total}</div>
            <div style="color:#64748b; font-size:0.8em; text-transform:uppercase;">Total Analyses</div>
        </div>
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.2em; font-weight:800; color:#f43f5e;">{manip}</div>
            <div style="color:#64748b; font-size:0.8em; text-transform:uppercase;">Manipulative</div>
        </div>
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.2em; font-weight:800; color:#4ade80;">{neutral}</div>
            <div style="color:#64748b; font-size:0.8em; text-transform:uppercase;">Neutral</div>
        </div>
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.2em; font-weight:800; color:#eab308;">{avg_sev:.1f}</div>
            <div style="color:#64748b; font-size:0.8em; text-transform:uppercase;">Avg Severity</div>
        </div>
    </div>
    """

    # -- Technique frequency bars --
    tech_freq = stats.get("technique_frequency", {})
    if tech_freq:
        max_count = max(tech_freq.values()) if tech_freq else 1
        html += '<div class="glass-card" style="margin-bottom:16px;"><h4 style="color:#f1f5f9; margin:0 0 16px;">Most Detected Techniques</h4>'
        for tech, count in list(tech_freq.items())[:10]:
            pct = int((count / max_count) * 100)
            color = TECHNIQUE_COLORS.get(tech, "#666")
            label = tech.replace("_", " ").title()
            html += f"""
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span style="color:#cbd5e1; font-size:0.85em;">{_esc(label)}</span>
                    <span style="color:{color}; font-weight:600; font-size:0.85em;">{count}x</span>
                </div>
                <div style="background:rgba(255,255,255,0.04); border-radius:4px; height:6px; overflow:hidden;">
                    <div style="width:{pct}%; background:{color}; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            """
        html += "</div>"

    # -- Source breakdown --
    source_b = stats.get("source_breakdown", {})
    if source_b:
        html += '<div class="glass-card" style="margin-bottom:16px;"><h4 style="color:#f1f5f9; margin:0 0 12px;">Analysis Sources</h4><div style="display:flex; gap:12px; flex-wrap:wrap;">'
        source_icons = {"text": "&#128196;", "url": "&#127760;", "file": "&#128193;", "batch": "&#128218;"}
        for src, count in source_b.items():
            icon = source_icons.get(src, "&#128196;")
            html += f'<div style="padding:10px 16px; background:rgba(255,255,255,0.04); border-radius:10px; text-align:center;"><div style="font-size:1.5em;">{icon}</div><div style="color:#f1f5f9; font-weight:700;">{count}</div><div style="color:#64748b; font-size:0.8em;">{_esc(src.title())}</div></div>'
        html += "</div></div>"

    # -- Recent analyses --
    recent = stats.get("recent", [])
    if recent:
        html += '<div class="glass-card"><h4 style="color:#f1f5f9; margin:0 0 12px;">Recent Analyses</h4>'
        for entry in recent[:8]:
            label = entry.get("source_label", "")[:50]
            is_manip = entry.get("is_persuasive", False)
            score = int(entry.get("manipulation_score", 0) * 100)
            ts = entry.get("timestamp", "")[:16].replace("T", " ")
            color = "#f43f5e" if is_manip else "#4ade80"
            html += f"""
            <div style="display:flex; align-items:center; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#e2e8f0; font-size:0.9em; max-width:250px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{_esc(label)}</span>
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="color:#64748b; font-size:0.8em;">{ts}</span>
                    <span style="padding:2px 8px; border-radius:8px; background:{color}20; color:{color}; font-size:0.8em; font-weight:600;">{score}%</span>
                </div>
            </div>
            """
        html += "</div>"

    return html


# ---------------------------------------------------------------------------
# 10. Intelligence Center
# ---------------------------------------------------------------------------

def build_intelligence_html(report: dict) -> str:
    """Render the analyst intelligence workspace."""
    total = report.get("total_records", 0)
    if total == 0:
        return """
        <div class="glass-card" style="text-align:center; padding:50px;">
            <div style="font-size:3em; margin-bottom:16px;">&#128300;</div>
            <h2 style="color:#f1f5f9; margin:0 0 8px;">No Intelligence Data Yet</h2>
            <p style="color:#94a3b8;">Run several analyses first, then refresh this panel.</p>
        </div>
        """

    persuasive_rate = report.get("persuasive_rate", 0) * 100
    high_risk = report.get("high_risk_count", 0)
    avg_score = report.get("avg_manipulation_score", 0) * 100
    drift = report.get("drift", {})
    drift_status = drift.get("status", "unknown").replace("_", " ").title()
    status_color = {
        "Stable": "#22c55e",
        "Attention": "#f59e0b",
        "Drift Detected": "#ef4444",
    }.get(drift_status, "#94a3b8")

    html = f"""
    <div class="section-header">
        <div>
            <h3>&#128300; Analyst Intelligence</h3>
            <p>Operational signals from stored analysis history.</p>
        </div>
        <span class="section-pill">{_esc(drift_status)}</span>
    </div>
    <div class="intel-grid">
        <div class="intel-card">
            <div class="intel-kpi">{total}</div>
            <div class="intel-label">Records</div>
        </div>
        <div class="intel-card">
            <div class="intel-kpi" style="color:#f59e0b;">{persuasive_rate:.0f}%</div>
            <div class="intel-label">Persuasive Rate</div>
        </div>
        <div class="intel-card">
            <div class="intel-kpi" style="color:#ef4444;">{high_risk}</div>
            <div class="intel-label">High Risk</div>
        </div>
        <div class="intel-card">
            <div class="intel-kpi" style="color:#a5b4fc;">{avg_score:.0f}%</div>
            <div class="intel-label">Avg Score</div>
        </div>
    </div>
    """

    top_techniques = report.get("top_techniques", {})
    if top_techniques:
        max_count = max(top_techniques.values()) or 1
        html += '<div class="glass-card"><h4 style="color:#f1f5f9;margin:0 0 14px;">Top Techniques</h4>'
        for tech, count in list(top_techniques.items())[:8]:
            color = TECHNIQUE_COLORS.get(tech, "#818cf8")
            pct = int(count / max_count * 100)
            html += f"""
            <div style="margin:9px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#cbd5e1;">{_esc(tech.replace('_', ' ').title())}</span>
                    <span style="color:{color};font-weight:700;">{count}</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:8px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:{color};"></div>
                </div>
            </div>
            """
        html += "</div>"

    active = report.get("active_learning", {}).get("items", [])
    html += '<div class="glass-card"><h4 style="color:#f1f5f9;margin:0 0 14px;">Active Learning Queue</h4>'
    if active:
        for item in active:
            reasons = ", ".join(item.get("reasons", []))
            html += f"""
            <div style="border-bottom:1px solid rgba(255,255,255,0.06);padding:10px 0;">
                <div style="display:flex;justify-content:space-between;gap:12px;">
                    <strong style="color:#e2e8f0;">Review #{item.get('analysis_id', 0)}</strong>
                    <span style="color:#f59e0b;font-weight:800;">{item.get('uncertainty_score', 0):.2f}</span>
                </div>
                <p style="color:#94a3b8;margin:5px 0 0;">{_esc(item.get('text_preview', ''))}</p>
                <p style="color:#64748b;margin:5px 0 0;font-size:0.82em;">{_esc(reasons)}</p>
            </div>
            """
    else:
        html += '<p style="color:#94a3b8;margin:0;">No uncertain examples found yet.</p>'
    html += "</div>"

    clusters = report.get("narrative_clusters", {}).get("items", [])
    html += '<div class="glass-card"><h4 style="color:#f1f5f9;margin:0 0 14px;">Narrative Clusters</h4>'
    if clusters:
        for cluster in clusters:
            html += f"""
            <div style="background:rgba(15,23,42,0.48);border:1px solid rgba(148,163,184,0.1);border-radius:8px;padding:12px;margin:8px 0;">
                <div style="display:flex;justify-content:space-between;gap:12px;">
                    <strong style="color:#a5b4fc;">{_esc(cluster.get('signature', ''))}</strong>
                    <span style="color:#f8fafc;">{cluster.get('size', 0)} items</span>
                </div>
                <div style="color:#94a3b8;font-size:0.86em;margin-top:4px;">Avg score: {cluster.get('avg_manipulation_score', 0) * 100:.0f}%</div>
            </div>
            """
    else:
        html += '<p style="color:#94a3b8;margin:0;">No repeated narrative clusters yet.</p>'
    html += "</div>"

    html += f"""
    <div class="glass-card">
        <h4 style="color:#f1f5f9;margin:0 0 14px;">Drift Monitor</h4>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            <div class="intel-card"><div class="intel-label">Status</div><div style="color:{status_color};font-weight:800;">{_esc(drift_status)}</div></div>
            <div class="intel-card"><div class="intel-label">Technique JS</div><div style="color:#f8fafc;font-weight:800;">{drift.get('technique_js_divergence', 0)}</div></div>
            <div class="intel-card"><div class="intel-label">Language JS</div><div style="color:#f8fafc;font-weight:800;">{drift.get('language_js_divergence', 0)}</div></div>
            <div class="intel-card"><div class="intel-label">Score Shift</div><div style="color:#f8fafc;font-weight:800;">{drift.get('score_shift', 0)}</div></div>
        </div>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# 11. Education Center
# ---------------------------------------------------------------------------

EDUCATION_CONTENT: dict[str, dict] = {
    "appeal_to_fear": {
        "name": "Appeal to Fear",
        "danger": 4,
        "description": "Manipulates the audience by presenting a scary scenario to push them toward a particular action or belief.",
        "how_to_spot": "Look for catastrophic predictions, worst-case scenarios presented as certainties, and urgent language like 'if we don't act now...'",
        "example_en": "If we don't pass this law immediately, crime will destroy our neighborhoods and your family won't be safe.",
        "example_fr": "Si nous ne votons pas cette loi immediatement, la criminalite detruira nos quartiers.",
    },
    "appeal_to_authority": {
        "name": "Appeal to Authority",
        "danger": 3,
        "description": "Uses the opinion of an authority figure or institution to support a claim, without providing actual evidence.",
        "how_to_spot": "Watch for vague references like 'experts say', 'scientists agree', 'studies show' without citing specific sources.",
        "example_en": "Every top economist agrees that this policy is the only solution. You can't argue with the experts.",
        "example_fr": "Tous les experts sont d'accord que cette politique est la seule solution.",
    },
    "loaded_language": {
        "name": "Loaded Language",
        "danger": 4,
        "description": "Uses emotionally charged words or phrases to influence opinion by triggering emotional reactions rather than rational thought.",
        "how_to_spot": "Look for adjectives/adverbs that carry strong emotional weight: 'devastating', 'heroic', 'disgusting', 'radical', 'catastrophic'.",
        "example_en": "The devastating new policy will ruthlessly destroy the livelihoods of hardworking families.",
        "example_fr": "Cette politique devastatrice va impitoyablement detruire les moyens de subsistance des familles.",
    },
    "false_dilemma": {
        "name": "False Dilemma",
        "danger": 3,
        "description": "Presents only two options (usually one good and one bad) when in reality there are many more possibilities.",
        "how_to_spot": "Look for 'either/or' framing, 'you're either with us or against us', 'there are only two choices'.",
        "example_en": "You're either with us in this fight, or you're part of the problem. There's no middle ground.",
        "example_fr": "Vous etes soit avec nous dans ce combat, soit vous faites partie du probleme.",
    },
    "bandwagon": {
        "name": "Bandwagon",
        "danger": 3,
        "description": "Pressures people to conform by implying that 'everyone' already agrees or is doing something.",
        "how_to_spot": "Watch for 'everyone knows', 'millions already', 'join the movement', 'don't be left behind'.",
        "example_en": "Millions of people have already switched. Don't be the last one left behind!",
        "example_fr": "Des millions de personnes ont deja change. Ne soyez pas le dernier !",
    },
    "ad_hominem": {
        "name": "Ad Hominem",
        "danger": 3,
        "description": "Attacks the person making the argument rather than addressing the argument itself.",
        "how_to_spot": "Look for personal insults, character attacks, or irrelevant personal details used to discredit someone's position.",
        "example_en": "Why should we listen to him? He's a college dropout who can't even manage his own finances.",
        "example_fr": "Pourquoi l'ecouter ? C'est un decrocheur qui ne sait meme pas gerer ses finances.",
    },
    "exaggeration": {
        "name": "Exaggeration / Hyperbole",
        "danger": 3,
        "description": "Overstates facts, statistics, or consequences to make an argument seem more compelling than it actually is.",
        "how_to_spot": "Look for absolute terms: 'always', 'never', 'everyone', 'nobody', 'the worst ever', 'unprecedented crisis'.",
        "example_en": "This is the greatest crisis in the entire history of humanity. Nothing has ever been this bad.",
        "example_fr": "C'est la plus grande crise de toute l'histoire de l'humanite. Rien n'a jamais ete aussi grave.",
    },
    "whataboutism": {
        "name": "Whataboutism",
        "danger": 3,
        "description": "Deflects criticism by pointing to someone else's faults instead of addressing the actual issue.",
        "how_to_spot": "Watch for 'but what about...', 'they did it too', 'look at what X did', used to avoid answering criticism.",
        "example_en": "Sure our company pollutes, but what about the other companies? They're even worse!",
        "example_fr": "Bien sur notre entreprise pollue, mais qu'en est-il des autres ? Ils sont pires !",
    },
}


def build_education_html() -> str:
    """Build the education center content."""
    html = """
    <div class="glass-card" style="margin-bottom:20px;">
        <h3 style="color:#f1f5f9; margin:0 0 8px;">&#127891; Learn to Recognize Manipulation</h3>
        <p style="color:#94a3b8; margin:0; font-size:0.95em;">
            Understanding persuasion techniques is your best defense against manipulation.
            Below are the most common techniques with examples and tips for recognition.
        </p>
    </div>
    """

    for tech_id, data in EDUCATION_CONTENT.items():
        color = TECHNIQUE_COLORS.get(tech_id, "#666")
        icon = TECHNIQUE_ICONS.get(tech_id, "")
        danger = data["danger"]
        danger_dots = f'<span style="color:#f43f5e;">{"&#9679; " * danger}</span><span style="color:rgba(255,255,255,0.1);">{"&#9679; " * (5 - danger)}</span>'

        html += f"""
        <div class="glass-card" style="margin-bottom:12px; border-left:4px solid {color};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:1.4em;">{icon}</span>
                    <h4 style="margin:0; color:#f1f5f9;">{_esc(data['name'])}</h4>
                </div>
                <div style="font-size:0.8em;">Danger: {danger_dots}</div>
            </div>
            <p style="color:#cbd5e1; margin:0 0 12px; line-height:1.6;">{_esc(data['description'])}</p>
            <div style="background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; margin-bottom:10px;">
                <strong style="color:#eab308; font-size:0.85em;">&#128269; HOW TO SPOT IT:</strong>
                <p style="color:#94a3b8; margin:6px 0 0; font-size:0.9em;">{_esc(data['how_to_spot'])}</p>
            </div>
            <div style="background:{color}10; padding:12px; border-radius:8px; border-left:3px solid {color};">
                <strong style="color:{color}; font-size:0.85em;">EXAMPLE:</strong>
                <p style="color:#e2e8f0; margin:6px 0 0; font-style:italic; font-size:0.9em;">&ldquo;{_esc(data['example_en'])}&rdquo;</p>
            </div>
        </div>
        """

    return html
