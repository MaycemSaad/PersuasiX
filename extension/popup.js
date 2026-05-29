/**
 * PersuasiX Browser Extension — Popup Script
 *
 * Handles:
 *  - Text analysis via API
 *  - Current page analysis
 *  - Selected text analysis
 *  - Result display with scores and techniques
 *  - Communication with content script for highlighting
 */

// ---------------------------------------------------------------------------
// Config & State
// ---------------------------------------------------------------------------

const DEFAULT_API_URL = 'http://localhost:8000';
let apiUrl = DEFAULT_API_URL;
let isConnected = false;
let lastResult = null;

// DOM elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const analyzePageBtn = document.getElementById('analyzePageBtn');
const analyzeSelectionBtn = document.getElementById('analyzeSelectionBtn');
const langSelect = document.getElementById('langSelect');
const autoHighlight = document.getElementById('autoHighlight');
const inputSection = document.getElementById('inputSection');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const scoreCircle = document.getElementById('scoreCircle');
const verdictText = document.getElementById('verdictText');
const techniquesList = document.getElementById('techniquesList');
const phrasesContainer = document.getElementById('phrasesContainer');
const highlightPageBtn = document.getElementById('highlightPageBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  // Load saved settings
  const stored = await chrome.storage.local.get(['apiUrl', 'language', 'autoHighlight']);
  if (stored.apiUrl) apiUrl = stored.apiUrl;
  if (stored.language) langSelect.value = stored.language;
  if (stored.autoHighlight !== undefined) autoHighlight.checked = stored.autoHighlight;

  // Check API connection
  await checkApiConnection();

  // Event listeners
  analyzeBtn.addEventListener('click', analyzeText);
  analyzePageBtn.addEventListener('click', analyzePage);
  analyzeSelectionBtn.addEventListener('click', analyzeSelection);
  highlightPageBtn.addEventListener('click', highlightOnPage);
  newAnalysisBtn.addEventListener('click', resetUI);

  langSelect.addEventListener('change', () => {
    chrome.storage.local.set({ language: langSelect.value });
  });

  autoHighlight.addEventListener('change', () => {
    chrome.storage.local.set({ autoHighlight: autoHighlight.checked });
  });

  textInput.addEventListener('input', () => {
    analyzeBtn.disabled = !textInput.value.trim();
  });
});

// ---------------------------------------------------------------------------
// API Communication
// ---------------------------------------------------------------------------

async function checkApiConnection() {
  try {
    const response = await fetch(`${apiUrl}/health`, { method: 'GET', signal: AbortSignal.timeout(5000) });
    if (response.ok) {
      const data = await response.json();
      isConnected = true;
      statusDot.classList.add('connected');
      statusText.textContent = `Connected to PersuasiX API (${data.pipeline_status || 'ready'})`;
      analyzeBtn.disabled = !textInput.value.trim();
    } else {
      throw new Error('API not healthy');
    }
  } catch (e) {
    isConnected = false;
    statusDot.classList.remove('connected');
    statusText.textContent = 'API offline — start with: uvicorn api.main:app';
    analyzeBtn.disabled = true;
  }
}

async function callAnalyzeAPI(text, language) {
  const response = await fetch(`${apiUrl}/api/v1/analyze/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text,
      language: language || 'en',
      threshold: 0.5,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API error: ${response.status} — ${error}`);
  }

  return await response.json();
}

// ---------------------------------------------------------------------------
// Analysis Functions
// ---------------------------------------------------------------------------

async function analyzeText() {
  const text = textInput.value.trim();
  if (!text || !isConnected) return;

  showLoading();

  try {
    const result = await callAnalyzeAPI(text, langSelect.value);
    lastResult = result;
    displayResult(result);
  } catch (e) {
    showError(e.message);
  }
}

async function analyzePage() {
  showLoading();

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [{ result: pageText }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        // Extract main content text
        const selectors = ['article', 'main', '[role="main"]', '.content', '.post-content', '#content'];
        for (const sel of selectors) {
          const el = document.querySelector(sel);
          if (el && el.innerText.trim().length > 100) return el.innerText.trim();
        }
        return document.body.innerText.trim();
      },
    });

    if (!pageText || pageText.length < 20) {
      showError('Could not extract text from this page.');
      return;
    }

    // Truncate to ~3000 words
    const words = pageText.split(/\s+/);
    const truncated = words.length > 3000 ? words.slice(0, 3000).join(' ') : pageText;

    const result = await callAnalyzeAPI(truncated, langSelect.value);
    lastResult = result;
    displayResult(result);

    // Auto-highlight if enabled
    if (autoHighlight.checked && result.highlighted_phrases?.length > 0) {
      highlightOnPage();
    }
  } catch (e) {
    showError(e.message);
  }
}

async function analyzeSelection() {
  showLoading();

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [{ result: selectedText }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    });

    if (!selectedText || selectedText.trim().length < 10) {
      showError('Please select some text on the page first.');
      return;
    }

    const result = await callAnalyzeAPI(selectedText.trim(), langSelect.value);
    lastResult = result;
    displayResult(result);
  } catch (e) {
    showError(e.message);
  }
}

// ---------------------------------------------------------------------------
// Highlight on Page
// ---------------------------------------------------------------------------

async function highlightOnPage() {
  if (!lastResult || !lastResult.highlighted_phrases?.length) return;

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: injectHighlights,
      args: [lastResult.highlighted_phrases],
    });
  } catch (e) {
    console.error('Highlight injection failed:', e);
  }
}

// This function runs in the page context
function injectHighlights(phrases) {
  // Remove previous highlights
  document.querySelectorAll('.persuasix-highlight').forEach(el => {
    el.replaceWith(el.textContent);
  });

  const techniqueColors = {
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
    default: '#818cf8',
  };

  // Walk text nodes and highlight matching phrases
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  for (const phrase of phrases) {
    const searchText = phrase.phrase;
    if (!searchText) continue;
    const color = techniqueColors[phrase.technique] || techniqueColors.default;

    for (const node of textNodes) {
      const idx = node.textContent.indexOf(searchText);
      if (idx === -1) continue;

      const before = node.textContent.substring(0, idx);
      const match = node.textContent.substring(idx, idx + searchText.length);
      const after = node.textContent.substring(idx + searchText.length);

      const span = document.createElement('span');
      span.className = 'persuasix-highlight';
      span.style.cssText = `
        background: ${color}33;
        border-bottom: 2px solid ${color};
        padding: 1px 2px;
        border-radius: 3px;
        cursor: help;
        position: relative;
      `;
      span.title = `[${phrase.technique.replace(/_/g, ' ')}] ${phrase.evidence || ''}`;
      span.textContent = match;

      const parent = node.parentNode;
      if (before) parent.insertBefore(document.createTextNode(before), node);
      parent.insertBefore(span, node);
      if (after) parent.insertBefore(document.createTextNode(after), node);
      parent.removeChild(node);
      break; // One replacement per phrase per pass
    }
  }
}

// ---------------------------------------------------------------------------
// UI Display
// ---------------------------------------------------------------------------

function displayResult(result) {
  hideLoading();
  inputSection.style.display = 'none';
  resultSection.classList.add('visible');

  // Score
  const score = Math.round((result.manipulation_score || 0) * 100);
  scoreCircle.textContent = score;
  scoreCircle.className = 'score-circle';

  if (score < 30) {
    scoreCircle.classList.add('safe');
    verdictText.className = 'verdict safe';
    verdictText.textContent = 'Neutral / Low Risk';
  } else if (score < 60) {
    scoreCircle.classList.add('warning');
    verdictText.className = 'verdict warning';
    verdictText.textContent = 'Moderately Manipulative';
  } else {
    scoreCircle.classList.add('danger');
    verdictText.className = 'verdict danger';
    verdictText.textContent = 'Highly Manipulative';
  }

  // Techniques
  techniquesList.innerHTML = '';
  const techniques = result.techniques || [];
  const probs = result.technique_probabilities || {};

  if (techniques.length === 0) {
    techniquesList.innerHTML = '<li style="border-left-color: var(--success);">No manipulation techniques detected</li>';
  } else {
    for (const tech of techniques) {
      const confidence = Math.round((probs[tech] || 0.8) * 100);
      const li = document.createElement('li');
      li.innerHTML = `
        <span class="tech-name">${tech.replace(/_/g, ' ')}</span>
        <span class="tech-confidence">${confidence}%</span>
      `;
      techniquesList.appendChild(li);
    }
  }

  // Highlighted phrases
  phrasesContainer.innerHTML = '';
  const phrases = result.highlighted_phrases || [];
  if (phrases.length === 0) {
    phrasesContainer.innerHTML = '<p>No manipulative phrases identified.</p>';
  } else {
    for (const phrase of phrases.slice(0, 5)) {
      const div = document.createElement('div');
      div.style.cssText = 'margin-bottom: 8px; padding: 6px 8px; background: rgba(15,23,42,0.4); border-radius: 6px; border-left: 2px solid var(--danger);';
      div.innerHTML = `
        <div style="color: var(--danger); font-weight: 600; font-size: 11px; margin-bottom: 2px;">[${(phrase.technique || '').replace(/_/g, ' ')}]</div>
        <div style="color: var(--text-primary); font-style: italic;">"${escapeHtml(phrase.phrase || '')}"</div>
        ${phrase.evidence ? `<div style="color: var(--text-secondary); margin-top: 3px; font-size: 11px;">${escapeHtml(phrase.evidence)}</div>` : ''}
      `;
      phrasesContainer.appendChild(div);
    }
  }
}

function showLoading() {
  inputSection.style.display = 'none';
  resultSection.classList.remove('visible');
  loadingSection.classList.add('visible');
}

function hideLoading() {
  loadingSection.classList.remove('visible');
}

function showError(message) {
  hideLoading();
  inputSection.style.display = 'block';
  alert(`PersuasiX Error: ${message}`);
}

function resetUI() {
  resultSection.classList.remove('visible');
  inputSection.style.display = 'block';
  lastResult = null;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
