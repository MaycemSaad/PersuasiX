# Publishing PersuasiX Chrome Extension to Chrome Web Store

## Prerequisites

1. **Google Developer Account** — Register at https://chrome.google.com/webstore/devconsole ($5 one-time fee)
2. **Extension icons** — Already generated in `icons/` (16, 48, 128px)
3. **Working API** — The extension requires the PersuasiX API running

## Step 1: Prepare the Package

```bash
# From the extension/ directory
cd extension/

# Generate icons if needed
python icons/generate_icons.py

# Create the ZIP package for upload
zip -r persuasix-extension.zip \
  manifest.json \
  popup.html popup.js \
  background.js \
  content.js content.css \
  options.html \
  icons/
```

## Step 2: Chrome Web Store Listing

### Required Information

| Field | Value |
|---|---|
| **Name** | PersuasiX — Persuasion Detector |
| **Summary** | Real-time detection of persuasion and manipulation techniques in web content |
| **Category** | Productivity |
| **Language** | English |

### Description (for Store listing)

```
PersuasiX detects persuasion and manipulation techniques in any web page in real-time.

FEATURES:
- Analyze any web page for 18 persuasion techniques
- Highlight manipulative phrases directly on the page
- Right-click to analyze selected text
- Support for 7 languages (English, French, Arabic, Spanish, German, Chinese, Hindi)
- Color-coded technique identification with explanations
- Manipulation risk score (0-100)

HOW IT WORKS:
1. Click the PersuasiX icon or right-click selected text
2. The text is analyzed by our NLP engine
3. Manipulative phrases are highlighted with technique labels
4. A risk score helps you assess content credibility

SUPPORTED TECHNIQUES:
Appeal to Fear, Loaded Language, Bandwagon, False Dilemma,
Ad Hominem, Appeal to Authority, Whataboutism, Exaggeration,
Name Calling, Flag Waving, and 8 more.

PRIVACY:
- No data collected or stored on external servers
- Analysis runs through your own local API server
- No tracking, no ads, no third-party data sharing

REQUIREMENTS:
- PersuasiX API server running locally (see GitHub for setup)
- Or connect to a hosted PersuasiX API instance

Open source: https://github.com/MaycemSaad/PersuasiX
```

### Screenshots Needed
1. Extension popup showing analysis results
2. In-page highlighting of manipulative phrases
3. Right-click context menu
4. Settings/options page

## Step 3: Upload to Chrome Web Store

1. Go to https://chrome.google.com/webstore/devconsole
2. Click "New Item"
3. Upload `persuasix-extension.zip`
4. Fill in the listing information above
5. Upload screenshots (1280x800 or 640x400)
6. Set visibility (Public or Unlisted for testing)
7. Submit for review

## Step 4: Review Process

- Google reviews typically take 1-3 business days
- Common rejection reasons:
  - Missing privacy policy
  - Excessive permissions
  - Misleading description
- Our extension uses minimal permissions (activeTab, contextMenus, storage)

## Privacy Policy

Host a privacy policy page stating:
- No personal data is collected
- Text analysis is performed via user's own API server
- No data is sent to third-party services
- Extension stores only user preferences locally

## Updating the Extension

1. Increment version in `manifest.json`
2. Create new ZIP
3. Upload to Developer Dashboard
4. Submit for re-review
