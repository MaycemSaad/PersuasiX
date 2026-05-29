/**
 * PersuasiX Browser Extension — Background Service Worker
 *
 * Handles:
 *  - Context menu for right-click "Analyze with PersuasiX"
 *  - Badge updates showing page risk level
 *  - Message passing between popup and content scripts
 */

const API_URL = 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Context Menu
// ---------------------------------------------------------------------------

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'persuasix-analyze',
    title: 'Analyze with PersuasiX',
    contexts: ['selection'],
  });

  chrome.contextMenus.create({
    id: 'persuasix-analyze-page',
    title: 'Analyze Page with PersuasiX',
    contexts: ['page'],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'persuasix-analyze' && info.selectionText) {
    try {
      const result = await analyzeText(info.selectionText);
      updateBadge(tab.id, result);

      // Send result to content script for inline display
      chrome.tabs.sendMessage(tab.id, {
        type: 'PERSUASIX_RESULT',
        data: result,
        selection: info.selectionText,
      });
    } catch (e) {
      console.error('Analysis failed:', e);
    }
  }

  if (info.menuItemId === 'persuasix-analyze-page') {
    try {
      const [{ result: pageText }] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          const selectors = ['article', 'main', '[role="main"]', '.content'];
          for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText.trim().length > 100) return el.innerText.trim();
          }
          return document.body.innerText.substring(0, 10000);
        },
      });

      if (pageText) {
        const words = pageText.split(/\s+/);
        const truncated = words.length > 3000 ? words.slice(0, 3000).join(' ') : pageText;
        const result = await analyzeText(truncated);
        updateBadge(tab.id, result);

        chrome.tabs.sendMessage(tab.id, {
          type: 'PERSUASIX_PAGE_RESULT',
          data: result,
        });
      }
    } catch (e) {
      console.error('Page analysis failed:', e);
    }
  }
});

// ---------------------------------------------------------------------------
// API Call
// ---------------------------------------------------------------------------

async function analyzeText(text) {
  const response = await fetch(`${API_URL}/api/v1/analyze/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, language: 'en', threshold: 0.5 }),
  });

  if (!response.ok) throw new Error(`API returned ${response.status}`);
  return response.json();
}

// ---------------------------------------------------------------------------
// Badge
// ---------------------------------------------------------------------------

function updateBadge(tabId, result) {
  const score = Math.round((result.manipulation_score || 0) * 100);
  let color, text;

  if (score < 30) {
    color = '#22c55e'; text = 'OK';
  } else if (score < 60) {
    color = '#f59e0b'; text = score.toString();
  } else {
    color = '#ef4444'; text = score.toString();
  }

  chrome.action.setBadgeText({ text, tabId });
  chrome.action.setBadgeBackgroundColor({ color, tabId });
}

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'ANALYZE_REQUEST') {
    analyzeText(message.text)
      .then(result => sendResponse({ success: true, data: result }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true; // async response
  }

  if (message.type === 'CHECK_API') {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(data => sendResponse({ connected: true, data }))
      .catch(() => sendResponse({ connected: false }));
    return true;
  }
});
