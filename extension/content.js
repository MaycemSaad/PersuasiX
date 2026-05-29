/**
 * PersuasiX Browser Extension — Content Script
 *
 * Runs on every page to:
 *  - Listen for analysis results from background/popup
 *  - Inject highlight overlays on manipulative text
 *  - Show floating tooltip with technique details
 */

// ---------------------------------------------------------------------------
// Message listener
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'PERSUASIX_RESULT') {
    showInlineResult(message.data, message.selection);
  }

  if (message.type === 'PERSUASIX_PAGE_RESULT') {
    highlightPhrases(message.data.highlighted_phrases || []);
    showFloatingBanner(message.data);
  }

  if (message.type === 'HIGHLIGHT_PHRASES') {
    highlightPhrases(message.phrases);
  }
});

// ---------------------------------------------------------------------------
// Highlight manipulative phrases on page
// ---------------------------------------------------------------------------

const TECHNIQUE_COLORS = {
  appeal_to_fear: '#ef4444',
  loaded_language: '#f97316',
  appeal_to_emotion: '#ec4899',
  bandwagon: '#8b5cf6',
  false_dilemma: '#06b6d4',
  ad_hominem: '#f43f5e',
  appeal_to_authority: '#a855f7',
  exaggeration: '#eab308',
  doubt: '#64748b',
  whataboutism: '#14b8a6',
  name_calling: '#e11d48',
  flag_waving: '#3b82f6',
  straw_man: '#f472b6',
  red_herring: '#fb923c',
  causal_oversimplification: '#34d399',
  repetition: '#c084fc',
  slogans: '#fbbf24',
  thought_terminating_cliche: '#9ca3af',
};

function highlightPhrases(phrases) {
  // Remove existing highlights
  removeHighlights();

  if (!phrases || phrases.length === 0) return;

  // Inject styles if not already present
  if (!document.getElementById('persuasix-styles')) {
    const style = document.createElement('style');
    style.id = 'persuasix-styles';
    style.textContent = `
      .persuasix-highlight {
        position: relative;
        cursor: help;
        transition: all 0.2s;
      }
      .persuasix-highlight:hover {
        filter: brightness(1.2);
      }
      .persuasix-tooltip {
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: #0f172a;
        color: #f1f5f9;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 12px;
        line-height: 1.4;
        max-width: 300px;
        min-width: 200px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        border: 1px solid rgba(129, 140, 248, 0.3);
        z-index: 999999;
        display: none;
        pointer-events: none;
      }
      .persuasix-highlight:hover .persuasix-tooltip {
        display: block;
      }
      .persuasix-tooltip-technique {
        font-weight: 700;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 1px;
        margin-bottom: 4px;
      }
      .persuasix-tooltip-evidence {
        color: #94a3b8;
      }
    `;
    document.head.appendChild(style);
  }

  // Walk text nodes and highlight
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      if (node.parentElement.closest('.persuasix-highlight, .persuasix-banner, script, style, noscript')) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  for (const phrase of phrases) {
    const searchText = phrase.phrase;
    if (!searchText || searchText.length < 3) continue;

    const color = TECHNIQUE_COLORS[phrase.technique] || '#818cf8';

    for (let i = 0; i < textNodes.length; i++) {
      const node = textNodes[i];
      const idx = node.textContent.indexOf(searchText);
      if (idx === -1) continue;

      const range = document.createRange();
      range.setStart(node, idx);
      range.setEnd(node, idx + searchText.length);

      const span = document.createElement('span');
      span.className = 'persuasix-highlight';
      span.style.background = `${color}25`;
      span.style.borderBottom = `2px solid ${color}`;
      span.style.padding = '1px 2px';
      span.style.borderRadius = '3px';

      // Tooltip
      const tooltip = document.createElement('div');
      tooltip.className = 'persuasix-tooltip';
      tooltip.innerHTML = `
        <div class="persuasix-tooltip-technique" style="color: ${color};">
          ${(phrase.technique || '').replace(/_/g, ' ')}
        </div>
        <div class="persuasix-tooltip-evidence">
          ${phrase.evidence || 'Manipulative phrase detected'}
        </div>
      `;

      range.surroundContents(span);
      span.appendChild(tooltip);
      break; // One highlight per phrase
    }
  }
}

function removeHighlights() {
  document.querySelectorAll('.persuasix-highlight').forEach(el => {
    const tooltip = el.querySelector('.persuasix-tooltip');
    if (tooltip) tooltip.remove();
    const parent = el.parentNode;
    parent.replaceChild(document.createTextNode(el.textContent), el);
    parent.normalize();
  });
}

// ---------------------------------------------------------------------------
// Floating banner for page-level results
// ---------------------------------------------------------------------------

function showFloatingBanner(result) {
  removeBanner();

  const score = Math.round((result.manipulation_score || 0) * 100);
  const techniques = result.techniques || [];
  const isManip = result.is_persuasive;

  let bgColor, borderColor, icon;
  if (score < 30) {
    bgColor = 'rgba(34, 197, 94, 0.95)'; borderColor = '#22c55e'; icon = '✅';
  } else if (score < 60) {
    bgColor = 'rgba(245, 158, 11, 0.95)'; borderColor = '#f59e0b'; icon = '⚠️';
  } else {
    bgColor = 'rgba(239, 68, 68, 0.95)'; borderColor = '#ef4444'; icon = '❌';
  }

  const banner = document.createElement('div');
  banner.className = 'persuasix-banner';
  banner.style.cssText = `
    position: fixed;
    top: 12px;
    right: 12px;
    background: #0f172a;
    color: #f1f5f9;
    padding: 14px 18px;
    border-radius: 12px;
    border: 1px solid ${borderColor};
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 13px;
    z-index: 999999;
    max-width: 320px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: opacity 0.3s;
  `;

  banner.innerHTML = `
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
      <span style="font-size: 18px;">${icon}</span>
      <div>
        <strong style="color: ${borderColor};">PersuasiX Score: ${score}/100</strong>
        <div style="font-size: 11px; color: #94a3b8;">
          ${isManip ? `${techniques.length} technique(s) detected` : 'No manipulation detected'}
        </div>
      </div>
      <button id="persuasix-close" style="
        margin-left: auto; background: none; border: none; color: #94a3b8;
        cursor: pointer; font-size: 16px; padding: 2px 6px;
      ">&times;</button>
    </div>
    ${techniques.length > 0 ? `
      <div style="font-size: 11px; color: #94a3b8;">
        ${techniques.map(t => `<span style="display: inline-block; background: rgba(129,140,248,0.15); color: #a5b4fc; padding: 2px 6px; border-radius: 4px; margin: 2px 2px 2px 0; font-size: 10px;">${t.replace(/_/g, ' ')}</span>`).join('')}
      </div>
    ` : ''}
  `;

  document.body.appendChild(banner);

  document.getElementById('persuasix-close').addEventListener('click', removeBanner);

  // Auto-dismiss after 15 seconds
  setTimeout(() => {
    if (banner.parentNode) {
      banner.style.opacity = '0';
      setTimeout(() => removeBanner(), 300);
    }
  }, 15000);
}

function removeBanner() {
  document.querySelectorAll('.persuasix-banner').forEach(el => el.remove());
}

// ---------------------------------------------------------------------------
// Inline result display (for selection analysis)
// ---------------------------------------------------------------------------

function showInlineResult(result, selectionText) {
  highlightPhrases(result.highlighted_phrases || []);
  showFloatingBanner(result);
}
