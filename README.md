# PersuasiX

### Multilingual Detection, Explanation & Neutralization of Persuasion Techniques in Text

> **PersuasiX** is a large-scale multilingual NLP platform that **detects**, **explains**, **neutralizes**, and **monitors** propaganda and manipulation techniques in real-time — across 7 languages, multiple media formats (text, audio, video), and social platforms (Mastodon, RSS, YouTube, Hacker News).

---

## Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
- [Architecture](#architecture)
- [The 18 Persuasion Techniques](#the-18-persuasion-techniques)
- [Dataset Pipeline](#dataset-pipeline)
- [The 4 NLP Tasks](#the-4-nlp-tasks)
- [Models](#models)
- [Social Media Monitoring](#social-media-monitoring)
- [Speech-to-Text Analysis](#speech-to-text-analysis)
- [Chrome Browser Extension](#chrome-browser-extension)
- [Collaborative Annotation](#collaborative-annotation)
- [Operational Intelligence](#operational-intelligence)
- [Model Distillation & Edge Deployment](#model-distillation--edge-deployment)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo App](#demo-app)
- [API Reference](#api-reference)
- [Results](#results)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

---

## Overview

Misinformation and manipulative rhetoric are pervasive in modern media — from news articles and political speeches to social media posts and video content. While existing NLP systems can *detect* propaganda, they fail at the crucial next steps: **explaining** the manipulation mechanism, **providing a neutral alternative**, and **monitoring content at scale in real-time**.

**PersuasiX** bridges this gap with a comprehensive platform:

```
                    ┌─────────────────────────────────────────────────────┐
                    │              PersuasiX Platform                      │
                    ├─────────────────────────────────────────────────────┤
                    │                                                     │
 Input Sources      │   Core NLP Pipeline        Output & Actions         │
 ─────────────      │   ────────────────         ────────────────         │
 • Text             │                                                     │
 • URLs/Articles    │   Detection ──┐            • Technique Labels       │
 • Audio/Video ─────┤   Explanation ├──────────► • Explanations (7 lang)  │
 • RSS Feeds        │   Neutralization           • Neutral Rewrites       │
 • Mastodon         │   Severity Scoring         • Risk Scores            │
 • YouTube          │   Span Detection           • Highlighted Spans      │
 • Hacker News      │   Fact-Checking            • Fact-Check Reports     │
                    │                            • PDF Reports            │
                    │                            • Alerts & Dashboards    │
                    └─────────────────────────────────────────────────────┘
```

| Feature | PersuasiX |
|---|---|
| **Languages** | English, French, Arabic, Spanish, German, Chinese, Hindi |
| **Techniques** | 18 fine-grained propaganda categories |
| **NLP Tasks** | Detection, Explanation, Neutralization, Scoring, Fact-Checking, Span Detection |
| **Training** | LoRA/QLoRA fine-tuning (4-bit quantization, 0.5-2% parameters) |
| **Social Monitoring** | Mastodon (real-time), RSS (17+ global feeds), YouTube Transcripts, Hacker News |
| **Speech Analysis** | Audio/video transcription via Whisper + timestamped persuasion analysis |
| **Browser Extension** | Chrome Manifest V3 with real-time in-page highlighting |
| **Annotation** | Collaborative labeling with inter-annotator agreement and export |
| **Intelligence** | Active learning, drift detection, narrative clustering, threat reports |
| **Distillation** | Knowledge distillation + ONNX export for edge/mobile deployment |
| **API** | FastAPI REST API with 20+ endpoints + Swagger docs |
| **Reports** | Professional PDF report generation |
| **Demo** | Interactive 9-tab Gradio app + HuggingFace Spaces (GPU) |
| **Robustness** | Tested against 12+ adversarial attack types |

---

## Key Innovation

Most existing work stops at **detection** (binary or multi-label classification). PersuasiX introduces eleven capabilities that make it a **complete media literacy and operational intelligence platform**:

1. **Explanation Generation** — Natural-language explanations of *why* each technique is manipulative, in 7 languages
2. **Text Neutralization** — Automatic rewriting that removes manipulation while preserving factual content
3. **Cross-lingual Severity Scoring** — Quantifying manipulation intensity using multilingual embeddings
4. **Span-level Detection** — Character-precise identification of manipulative phrases using RoBERTa+CRF hybrid with LLM validation
5. **Fact-Checking Integration** — Automatic claim extraction and verification via Google Fact Check Tools, ClaimBuster, and LLM-powered analysis
6. **Real-time Social Monitoring** — Free, no-API-key monitoring across Mastodon, RSS feeds, YouTube, and Hacker News with real-time streaming
7. **Speech-to-Text Pipeline** — Transcribe audio/video via Whisper, chunk by timestamp, analyze each segment for persuasion
8. **Browser Extension** — Chrome extension with in-page highlighting, context menu, and popup analysis
9. **Collaborative Annotation** — Multi-annotator task queues, span labels, inter-annotator agreement, and fine-tuning export
10. **Model Distillation** — Knowledge distillation to smaller models + ONNX/quantized export for edge deployment
11. **Operational Intelligence** — Active learning queues, distribution drift detection, narrative clustering, and automated threat reports

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PersuasiX Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  INPUT LAYER                                                                         │
│  ───────────                                                                         │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ Direct Text │ │  URL/Article │ │  Audio/Video │ │  Social    │ │   Chrome     │  │
│  │             │ │  Scraping    │ │  (Whisper)   │ │  Streams   │ │  Extension   │  │
│  └──────┬──────┘ └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ └──────┬───────┘  │
│         └───────────────┬┴───────────────┬┘               │               │          │
│                         ▼                ▼                 ▼               ▼          │
│  CORE NLP PIPELINE                                                                    │
│  ─────────────────                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                              │    │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐   │    │
│  │  │  RoBERTa     │   │  FLAN-T5     │   │  FLAN-T5     │   │  Sentence   │   │    │
│  │  │  Detector    │   │  Explainer   │   │  Neutralizer │   │  Transformer│   │    │
│  │  │  (18 labels) │   │  (7 langs)   │   │  (debiasing) │   │  (scoring)  │   │    │
│  │  └──────────────┘   └──────────────┘   └──────────────┘   └─────────────┘   │    │
│  │                                                                              │    │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │    │
│  │  │  Span        │   │  Fact        │   │  Severity    │                     │    │
│  │  │  Detector    │   │  Checker     │   │  Scorer      │                     │    │
│  │  │  (CRF+LLM)  │   │  (Multi-API) │   │  (0-5 scale) │                     │    │
│  │  └──────────────┘   └──────────────┘   └──────────────┘                     │    │
│  │                                                                              │    │
│  └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  OUTPUT LAYER                                                                         │
│  ────────────                                                                         │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐   │
│  │ REST API │ │ Gradio UI │ │ PDF Rpts │ │ Alerts   │ │ Annotation │ │ Intel    │   │
│  │ (20+ ep) │ │ (9 tabs)  │ │          │ │          │ │ Export     │ │ Reports  │   │
│  └──────────┘ └───────────┘ └──────────┘ └──────────┘ └────────────┘ └──────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## The 18 Persuasion Techniques

PersuasiX detects **18 fine-grained persuasion/propaganda techniques**, based on established rhetorical analysis frameworks:

| # | Technique | Description | Example |
|---|---|---|---|
| 1 | **Appeal to Fear** | Instills fear to influence decisions | *"If we don't act, catastrophe awaits"* |
| 2 | **Appeal to Authority** | Uses authority figures without evidence | *"Every scientist agrees"* |
| 3 | **Bandwagon** | Claims everyone agrees to pressure conformity | *"Everybody knows this is true"* |
| 4 | **False Dilemma** | Presents only two options when more exist | *"You're either with us or against us"* |
| 5 | **Ad Hominem** | Attacks the person, not the argument | *"He's a liar, don't listen to him"* |
| 6 | **Straw Man** | Misrepresents someone's argument | *"They want to destroy all tradition"* |
| 7 | **Red Herring** | Introduces irrelevant topics | *"Why talk about X when Y is the real issue?"* |
| 8 | **Loaded Language** | Uses emotionally charged words | *"This monstrous policy..."* |
| 9 | **Whataboutism** | Deflects by pointing to others' faults | *"What about when they did X?"* |
| 10 | **Causal Oversimplification** | Reduces complex issues to one cause | *"Regulation = fewer jobs. Simple."* |
| 11 | **Appeal to Emotion** | Exploits emotions over logic | *"Think of the crying children"* |
| 12 | **Repetition** | Repeats claims to make them seem true | *"Safe. Effective. Safe. Effective."* |
| 13 | **Exaggeration** | Overstates facts | *"This is our LAST chance"* |
| 14 | **Doubt** | Questions credibility without evidence | *"Can we really trust them?"* |
| 15 | **Slogans** | Catchy phrases replacing critical thinking | *"Make X great again"* |
| 16 | **Name Calling** | Labels opponents negatively | *"These radicals want to..."* |
| 17 | **Flag Waving** | Exploits patriotism | *"Our nation's heroes demand this"* |
| 18 | **Thought-Terminating Cliche** | Cliches that shut down debate | *"It is what it is"* |

---

## Dataset Pipeline

The dataset is built through a **4-stage reproducible pipeline**, scaling to **200K+ examples** across 7 languages:

```
Stage 1: Collection       →  Gather texts from corpora, news, social media, synthetic generation
Stage 2: Cleaning         →  Deduplicate, normalize, filter
Stage 3: LLM Enrichment   →  Generate explanations + neutral rewrites via GPT-4o-mini
Stage 4: Validation       →  Semantic similarity checks + quality scoring
```

### Stage Details

| Stage | Module | What it does |
|---|---|---|
| **Collection** | `src/data/collector.py` | Aggregates texts from SemEval corpora, news articles, social media, and synthetic generation |
| **Cleaning** | `src/data/cleaner.py` | Unicode normalization, URL removal, deduplication (MD5), length filtering |
| **Enrichment** | `src/data/enricher.py` | Uses GPT-4o-mini to generate: explanations (7 languages), neutral rewrites, severity scores, target audience |
| **Validation** | `src/data/validator.py` | Validates technique labels, checks explanation quality, computes semantic similarity |
| **Synthetic** | `scripts/generate_synthetic_data.py` | Template + LLM + paraphrase augmentation across all 7 languages |

### Dataset Statistics

| Metric | Value |
|---|---|
| Total examples | 200,000+ |
| Languages | 7 (EN, FR, AR, ES, DE, ZH, HI) |
| Persuasive examples | ~50% |
| Neutral examples | ~50% |
| Unique techniques | 18 |
| Avg. techniques per persuasive text | 2.3 |

---

## The 4 NLP Tasks

### Task 1: Persuasion Technique Detection
- **Model**: RoBERTa-base with custom classification head
- **Type**: Multi-label classification
- **Input**: Raw text
- **Output**: Binary vector over 18 techniques + probabilities
- **Loss**: BCEWithLogitsLoss
- **Training**: LoRA/QLoRA (4-bit) with rank 8-64, alpha 16-128

### Task 2: Explanation Generation
- **Model**: FLAN-T5-base (fine-tuned with LoRA)
- **Type**: Seq2Seq generation
- **Input**: Text + detected techniques
- **Output**: Natural-language explanation (in 7 languages)
- **Metrics**: ROUGE-L, BERTScore, BLEU

### Task 3: Text Neutralization
- **Model**: FLAN-T5-base (fine-tuned with LoRA)
- **Type**: Seq2Seq generation
- **Input**: Persuasive text + techniques to remove
- **Output**: Neutral, factual rewrite
- **Metrics**: ROUGE-L, semantic preservation score

### Task 4: Cross-lingual Severity Scoring
- **Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Type**: Embedding similarity
- **Input**: Original + neutral text pair
- **Output**: Manipulation score (0-1) + severity level (0-5)

### Span-level Detection
- **Model**: RoBERTa+CRF hybrid with LLM validation
- **Module**: `src/pipeline/span_detector.py`
- **Output**: Character-precise spans with technique labels
- **Approach**: BIO tagging with CRF layer + GPT-4o-mini cross-validation

### Fact-Checking
- **Module**: `src/pipeline/fact_checker.py`
- **APIs**: Google Fact Check Tools, ClaimBuster, LLM-powered verification
- **Output**: Claim extraction, verification status, source citations

---

## Models

| Component | Base Model | Parameters | Purpose |
|---|---|---|---|
| **Detector** | `roberta-base` | 125M | Multi-label technique classification |
| **Explainer** | `google/flan-t5-base` | 250M | Explanation generation |
| **Neutralizer** | `google/flan-t5-base` | 250M | Text debiasing / rewriting |
| **Scorer** | `paraphrase-multilingual-MiniLM-L12-v2` | 118M | Cross-lingual similarity |
| **Span Detector** | `roberta-base` + CRF | 125M | Token-level BIO tagging |
| **Distilled Student** | `distilbert-base` | 66M | Edge/mobile inference |

### LoRA/QLoRA Training

All seq2seq models are fine-tuned using **LoRA** (Low-Rank Adaptation) with optional **4-bit quantization**:

```python
# src/training/lora_trainer.py
LoRAConfig:
    rank: 8-64
    alpha: 16-128
    dropout: 0.05-0.1
    target_modules: ["q_proj", "v_proj"]
    quantization: 4-bit (QLoRA) or 8-bit
    trainable_params: 0.5-2% of total
```

### Detector Architecture

```
RoBERTa Encoder (12 layers, 768 hidden)
    ↓
[CLS] Token Pooling
    ↓
Dropout(0.1) → Linear(768, 256) → ReLU
    ↓
Dropout(0.1) → Linear(256, 18)
    ↓
BCEWithLogitsLoss (training) / Sigmoid (inference)
```

---

## Social Media Monitoring

PersuasiX includes a **real-time social media monitoring system** that fetches content from multiple platforms and runs persuasion analysis automatically — **all using free APIs with no keys required**.

### Supported Platforms

| Platform | Type | Auth Required | Real-time | Languages |
|----------|------|:---:|:---:|---|
| **Mastodon** | Fediverse | None | WebSocket streaming | en, fr, de, es, ar, ... |
| **RSS Feeds** | News (17+ outlets) | None | Polling (5min) | en, fr, ar, de, es |
| **YouTube Transcripts** | Video content | None | On-demand | 50+ languages |
| **Hacker News** | Tech/society debates | None | Firebase SSE | en |
| **Twitter/X** | Social media | Bearer Token (paid) | Filtered stream | multi |

### News Sources (RSS)

Pre-configured feeds from major global outlets:

- **English**: BBC World, BBC Politics, CNN, Fox News, Al Jazeera EN, NPR, The Guardian
- **French**: France24, Le Monde, RFI
- **Arabic**: Al Jazeera AR, BBC Arabic
- **German**: Deutsche Welle, Spiegel
- **Spanish**: BBC Mundo, El Pais

### Usage

```python
from src.pipeline.social_monitor import SocialMonitor

monitor = SocialMonitor()

# Real-time Mastodon streaming (FREE)
results = monitor.stream_mastodon(keywords=["propaganda", "manipulation"], duration=120)

# Fetch news from all RSS feeds (FREE)
news = monitor.fetch_rss_news(languages=["en", "fr", "ar"])

# Analyze a YouTube video transcript (FREE)
video_analysis = monitor.analyze_youtube_video("VIDEO_ID")

# Hacker News debates (FREE)
hn = monitor.fetch_hackernews(limit=20)

# Multi-platform search (FREE)
results = monitor.multi_platform_search("immigration", platforms=["mastodon", "rss", "hackernews"])

# Real-time monitoring with callbacks
def on_detection(result):
    if result.manipulation_score > 0.7:
        print(f"ALERT: {result.post.platform} — {result.techniques}")

monitor.monitor_realtime(keywords=["fear", "crisis"], duration=300, callback=on_detection)
```

### API Endpoints

```
GET  /api/v1/social/status          — Platform availability
POST /api/v1/social/mastodon/search — Search Mastodon
POST /api/v1/social/mastodon/stream — Stream real-time
GET  /api/v1/social/rss/fetch       — Fetch all RSS feeds
POST /api/v1/social/rss/search      — Search in feeds
POST /api/v1/social/youtube         — Analyze video transcript
GET  /api/v1/social/hackernews      — Fetch HN stories
POST /api/v1/social/multi-search    — Cross-platform search
```

---

## Speech-to-Text Analysis

PersuasiX analyzes **audio and video content** for persuasion techniques by transcribing speech and running the full NLP pipeline on timestamped segments.

### Pipeline

```
Audio/Video → Whisper Transcription → Timestamp Chunking → PersuasiX Analysis → Timeline Report
```

### Features

- **Dual transcription**: OpenAI Whisper API (cloud) or local Whisper model (offline, GPU)
- **Format support**: mp3, wav, m4a, ogg, flac, mp4, webm
- **Chunked analysis**: 60-second segments with individual scores
- **Timeline output**: Time-indexed manipulation scores for video scrubbing
- **Speaker diarization**: Optional speaker identification

### Usage

```python
from src.pipeline.speech_analyzer import SpeechAnalyzer

analyzer = SpeechAnalyzer(transcriber="auto")  # auto-selects API or local

# Analyze a local file
result = analyzer.analyze_file("political_speech.mp3", language="en")

print(f"Duration: {result.duration:.1f}s")
print(f"Overall manipulation score: {result.overall_score:.2%}")
print(f"Techniques found: {result.overall_techniques}")

# Timeline of manipulation intensity
for point in result.timeline:
    print(f"  [{point['timestamp']}] Score: {point['score']}% — {point['techniques']}")

# Analyze from URL
result = analyzer.analyze_url("https://example.com/speech.mp4")
```

### API Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/speech/file \
    -F "file=@speech.mp3" \
    -F "language=en"
```

---

## Chrome Browser Extension

A **Manifest V3 Chrome extension** that brings PersuasiX analysis directly into the browser, highlighting manipulative content in real-time as you browse.

### Features

- **Popup analysis**: Click the icon to analyze the current page
- **In-page highlighting**: Color-coded overlay on manipulative phrases
- **Context menu**: Right-click selected text → "Analyze with PersuasiX"
- **Badge notifications**: Red badge with manipulation score on the icon
- **Settings**: Configurable API endpoint, sensitivity, language
- **Technique tooltips**: Hover highlighted text for technique explanations

### Extension Structure

```
extension/
├── manifest.json          # Manifest V3 configuration
├── popup.html / popup.js  # Extension popup UI
├── background.js          # Service worker (context menu, badge)
├── content.js / content.css  # In-page highlighting & tooltips
├── options.html           # Settings page
└── icons/                 # 16px, 48px, 128px icons
```

### Installation (Developer Mode)

```bash
# Load unpacked in Chrome
1. Open chrome://extensions/
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the extension/ directory
```

### Publishing to Chrome Web Store

```bash
cd extension
python icons/generate_icons.py
zip -r persuasix-extension.zip manifest.json popup.html popup.js \
    background.js content.js content.css options.html icons/
# Upload to https://chrome.google.com/webstore/devconsole
```

Full publishing checklist in `extension/PUBLISHING.md`.

---

## Collaborative Annotation

A built-in **annotation interface** for building gold-standard datasets through multi-annotator labeling, enabling continuous model improvement.

### Features

- **Task queues**: Create and assign annotation tasks to multiple annotators
- **Span labeling**: Character-level span selection with technique labels
- **Document-level**: Binary persuasiveness, technique set, severity rating
- **Model feedback**: Agree/partial/disagree with model predictions
- **Inter-annotator agreement**: Automatic Jaccard similarity and verdict agreement
- **Export**: JSONL export with majority voting for fine-tuning

### Architecture

```
SQLite DB (annotations.db)
├── tasks table       — Texts to annotate (status tracking)
├── annotations table — Individual annotator labels
└── annotators table  — Annotator stats and agreement scores
```

### Usage

```python
from app.annotation import AnnotationDB, Annotation

db = AnnotationDB()

# Add tasks
db.add_task("text_001", "This is the only solution! Everyone agrees!", language="en")

# Submit annotation
annotation = Annotation(
    text_id="text_001",
    annotator="alice",
    is_persuasive=True,
    techniques=["Bandwagon", "False Dilemma"],
    severity=3.5,
    model_agreement="partial",
    spans=[{"start": 0, "end": 28, "technique": "False Dilemma"}],
)
db.save_annotation(annotation)

# Check agreement
agreement = db.compute_agreement("text_001")
# {"verdict_agreement": 1.0, "technique_jaccard": 0.75, ...}

# Export for training
db.export_for_training("data/annotated_export.jsonl", min_annotations=2)
```

### API Endpoints

```
POST /api/v1/annotate/tasks          — Create annotation task
POST /api/v1/annotate/tasks/bulk     — Bulk create tasks
GET  /api/v1/annotate/tasks/next     — Get next task for annotator
POST /api/v1/annotate/submit         — Submit annotation
GET  /api/v1/annotate/agreement/{id} — Compute inter-annotator agreement
GET  /api/v1/annotate/stats          — Annotation statistics
POST /api/v1/annotate/export         — Export for fine-tuning
```

---

## Operational Intelligence

An **analyst intelligence layer** that provides proactive insights from accumulated analysis data — identifying emerging narratives, detecting distribution shifts, and prioritizing uncertain examples for review.

### Capabilities

| Feature | Description |
|---------|-------------|
| **Active Learning** | Ranks uncertain predictions for human review to maximize labeling efficiency |
| **Drift Detection** | Compares recent technique distributions against baseline to catch emerging trends |
| **Narrative Clustering** | Groups similar manipulative texts to identify coordinated campaigns |
| **Threat Reports** | Automated reports combining risk scores, drift signals, and top threats |

### Usage

```python
from src.pipeline.intelligence import IntelligenceEngine

engine = IntelligenceEngine()

# Get uncertain examples for review
candidates = engine.get_active_learning_candidates(limit=20)

# Detect distribution drift
drift = engine.detect_drift(window_hours=24)
# {"drift_detected": True, "shifted_techniques": ["Appeal to Fear", ...]}

# Cluster narratives
clusters = engine.cluster_narratives(min_cluster_size=3)
# [{"theme": "immigration fear", "count": 12, "avg_score": 0.82}, ...]

# Generate threat report
report = engine.generate_threat_report()
```

### API Endpoints

```
GET /api/v1/intelligence/active-learning     — Uncertain examples for review
GET /api/v1/intelligence/drift               — Distribution drift analysis
GET /api/v1/intelligence/narrative-clusters   — Grouped narratives
GET /api/v1/intelligence/threat-report       — Combined threat assessment
```

---

## Model Distillation & Edge Deployment

PersuasiX includes a **knowledge distillation pipeline** for deploying lightweight models on edge devices (browsers, mobile, embedded systems).

### Pipeline

```
Teacher (RoBERTa-base, 125M) → Knowledge Distillation → Student (DistilBERT, 66M)
                                                              ↓
                                                        ONNX Export
                                                              ↓
                                                    ┌─────────────────────┐
                                                    │ • Dynamic quantized │
                                                    │ • INT8 optimized    │
                                                    │ • ~3x faster        │
                                                    │ • ~4x smaller       │
                                                    └─────────────────────┘
```

### Features

- **Teacher-student training**: Soft label distillation with temperature scaling
- **ONNX export**: Cross-platform inference (browser, mobile, C++)
- **Quantization**: Dynamic INT8 quantization for 4x size reduction
- **Benchmarking**: Automatic accuracy vs. speed comparison

### Usage

```bash
# Distill teacher to student
python -m src.pipeline.distiller \
    --teacher roberta-base \
    --student distilbert-base-uncased \
    --epochs 10 \
    --export-onnx \
    --quantize dynamic

# Output: models/distilled/
#   ├── student_model/          (PyTorch)
#   ├── student_model.onnx      (ONNX)
#   ├── student_quantized.onnx  (INT8)
#   └── benchmark_results.json
```

---

## Project Structure

```
persuasix/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package configuration
├── docker-compose.yml                 # Full stack deployment (API + Frontend)
├── config/
│   └── config.yaml                    # All hyperparameters and settings
├── data/
│   ├── raw/                           # Raw collected data
│   ├── processed/                     # Train/val/test splits
│   ├── synthetic/                     # Generated 200K+ dataset
│   └── sample/                        # Sample dataset for testing
├── src/
│   ├── data/
│   │   ├── collector.py               # Stage 1: Data collection (18 techniques)
│   │   ├── cleaner.py                 # Stage 2: Cleaning & normalization
│   │   ├── enricher.py                # Stage 3: LLM-based enrichment (7 languages)
│   │   ├── validator.py               # Stage 4: Quality validation
│   │   └── dataset.py                 # PyTorch Dataset classes
│   ├── models/
│   │   ├── detector.py                # RoBERTa multi-label classifier
│   │   ├── explainer.py               # FLAN-T5 explanation generator
│   │   ├── neutralizer.py             # FLAN-T5 text neutralizer
│   │   └── scorer.py                  # Multilingual similarity scorer
│   ├── training/
│   │   ├── trainer.py                 # Training loop (FP16, grad accum)
│   │   ├── lora_trainer.py            # LoRA/QLoRA fine-tuning (4-bit quantized)
│   │   ├── evaluator.py               # Multi-task evaluation
│   │   └── callbacks.py               # Early stopping, checkpointing
│   ├── pipeline/
│   │   ├── persuasix_pipeline.py      # End-to-end orchestration
│   │   ├── span_detector.py           # RoBERTa+CRF span-level detection
│   │   ├── fact_checker.py            # Multi-API fact verification
│   │   ├── social_monitor.py          # Mastodon, RSS, YouTube, HN monitoring
│   │   ├── speech_analyzer.py         # Whisper speech-to-text analysis
│   │   ├── intelligence.py            # Active learning, drift, clustering
│   │   └── distiller.py              # Distillation + ONNX edge export
│   └── utils/
│       ├── metrics.py                 # F1, ROUGE, BERTScore, BLEU
│       ├── visualization.py           # Plots and charts
│       └── helpers.py                 # Config, seed, device utilities
├── api/
│   ├── main.py                        # FastAPI app (20+ endpoints, Swagger)
│   ├── database.py                    # SQLAlchemy ORM (4 tables)
│   ├── schemas.py                     # Pydantic request/response models
│   └── routes/
│       ├── analysis.py                # Text/URL/File/Batch/FactCheck/Spans
│       ├── monitor.py                 # RSS feed monitoring endpoints
│       ├── intelligence.py            # Analyst intelligence endpoints
│       ├── social.py                  # Social platform monitoring endpoints
│       ├── speech.py                  # Audio/video analysis endpoints
│       └── reports.py                 # PDF report generation
├── app/
│   ├── app.py                         # Gradio 9-tab demo application
│   ├── components.py                  # 10+ HTML component builders
│   ├── scraper.py                     # URL scraping (trafilatura + BS4)
│   ├── file_parser.py                 # PDF/DOCX/TXT parsing
│   ├── annotation.py                  # Collaborative annotation system
│   └── history.py                     # JSON-based analysis history
├── deploy/
│   └── huggingface_space.py           # GPU-ready HuggingFace Spaces deployment
├── monitor/
│   └── rss_monitor.py                 # Automated RSS feed monitoring service
├── reports/
│   └── pdf_generator.py               # Professional PDF report generator
├── extension/                         # Chrome Browser Extension (Manifest V3)
│   ├── manifest.json                  # Extension configuration
│   ├── PUBLISHING.md                  # Chrome Web Store publishing guide
│   ├── popup.html / popup.js          # Extension popup UI
│   ├── background.js                  # Service worker (context menu, badges)
│   ├── content.js / content.css       # In-page highlighting & tooltips
│   ├── options.html                   # Extension settings page
│   └── icons/                         # Extension icons (16/48/128px)
├── scripts/
│   ├── generate_synthetic_data.py     # 200K+ dataset generation pipeline
│   ├── collect_data.py                # CLI: collect raw data
│   ├── build_dataset.py               # CLI: full dataset build pipeline
│   ├── train.py                       # CLI: train models
│   ├── evaluate.py                    # CLI: evaluate models
│   └── export_model.py               # CLI: export to ONNX/HuggingFace
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset analysis & visualization
│   ├── 02_model_training.ipynb        # Step-by-step training
│   └── 03_evaluation.ipynb            # Comprehensive evaluation
├── tests/
│   ├── test_data.py                   # Data pipeline tests
│   ├── test_models.py                 # Model architecture tests
│   ├── test_pipeline.py               # End-to-end pipeline tests
│   ├── test_social_monitor_live.py    # Live API integration tests
│   └── adversarial/
│       └── test_adversarial.py        # 12+ adversarial attack tests
└── docker/
    └── Dockerfile                     # Container for deployment
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (recommended for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/MaycemSaad/PersuasiX.git
cd PersuasiX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Environment Variables

```bash
# Required for dataset enrichment (optional for inference)
export OPENAI_API_KEY="your-key-here"

# Optional: Paid social media APIs
export TWITTER_BEARER_TOKEN="your-token"      # Twitter/X (paid)

# Optional: Speech-to-text (if using cloud Whisper)
export OPENAI_API_KEY="your-key"              # Same key as above

# Note: Mastodon, RSS, YouTube Transcripts, and Hacker News
# require NO API keys — they work out of the box.
```

---

## Quick Start

### 1. Build the dataset

```bash
# Offline mode (no API key needed — uses templates + synthetic generation)
python scripts/build_dataset.py --skip-enrichment

# Full mode (with GPT-4o-mini enrichment)
python scripts/build_dataset.py

# Generate 200K+ examples
python scripts/generate_synthetic_data.py --num-examples 200000 --languages en fr ar es de zh hi
```

### 2. Train all models

```bash
# Standard training
python scripts/train.py --task all

# LoRA fine-tuning (memory efficient)
python scripts/train.py --task all --lora --quantize 4bit
```

### 3. Evaluate

```bash
python scripts/evaluate.py --task all \
    --detector-path checkpoints/detector/final_model.pt \
    --explainer-path checkpoints/explainer/final_model \
    --neutralizer-path checkpoints/neutralizer/final_model
```

### 4. Launch the platform

```bash
# Gradio UI (9-tab studio)
python app/app.py
# Open http://localhost:7860

# FastAPI backend (20+ endpoints)
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

### 5. Start monitoring (no API keys needed)

```python
from src.pipeline.social_monitor import SocialMonitor

monitor = SocialMonitor()
results = monitor.multi_platform_search("propaganda", platforms=["mastodon", "rss", "hackernews"])
report = monitor.generate_report(results["mastodon"] + results["rss"])
print(f"Manipulation rate: {report['manipulation_rate']}%")
```

### Using the Pipeline in Code

```python
from src.pipeline.persuasix_pipeline import PersuasixPipeline

# Load with default models
pipeline = PersuasixPipeline.from_default_models(device="cpu")

# Analyze a text
result = pipeline.analyze(
    text="Only a fool would disagree. Everyone knows this is the only solution!",
    language="en",
)

print(f"Persuasive: {result.is_persuasive}")
print(f"Techniques: {result.techniques}")
print(f"Explanation: {result.explanation}")
print(f"Neutral version: {result.neutral_rewrite}")
print(f"Severity: {result.severity_score}/5")
print(f"Manipulation score: {result.manipulation_score:.2%}")
```

---

## Training

### Train specific models

```bash
# Detector only
python scripts/train.py --task detector

# Explainer only
python scripts/train.py --task explainer

# Neutralizer only
python scripts/train.py --task neutralizer

# All models
python scripts/train.py --task all

# With LoRA (4-bit quantized)
python scripts/train.py --task all --lora --quantize 4bit --rank 16 --alpha 32
```

### Hyperparameters

All hyperparameters are in `config/config.yaml`. Key settings:

| Parameter | Detector | Explainer | Neutralizer |
|---|---|---|---|
| Base model | RoBERTa-base | FLAN-T5-base | FLAN-T5-base |
| Learning rate | 2e-5 | 3e-5 | 3e-5 |
| Batch size | 16 | 8 | 8 |
| Epochs | 10 | 8 | 8 |
| Max seq length | 512 | 512 | 512 |
| LoRA rank | 16 | 8 | 8 |
| LoRA alpha | 32 | 16 | 16 |

### Training Features
- LoRA/QLoRA fine-tuning (4-bit quantization)
- Mixed-precision training (FP16)
- Gradient accumulation
- Cosine annealing learning rate schedule
- Early stopping (patience=3)
- Model checkpointing (best validation loss)

---

## Evaluation

### Metrics by Task

**Task 1 — Detection:**
- F1 (macro, micro, weighted)
- Precision / Recall (per technique)
- Hamming Loss
- Exact Match Ratio

**Task 2 & 3 — Generation:**
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU
- BERTScore (P, R, F1)

**Task 4 — Scoring:**
- Cosine similarity (original vs neutral)
- Cross-lingual consistency

**Adversarial Robustness:**
- 12+ attack types (homoglyphs, prompt injection, semantic perturbation, etc.)
- Robustness score per attack category
- Test suite: `tests/adversarial/test_adversarial.py`

### Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Demo App

The Gradio-powered **PersuasiX Studio** provides a 9-tab interface:

| Tab | Function |
|-----|----------|
| **Text Analysis** | Paste text, get full analysis with highlighting |
| **URL Analysis** | Scrape any webpage and analyze |
| **File Upload** | Analyze PDF, DOCX, TXT documents |
| **Compare** | Side-by-side original vs. neutralized |
| **Batch** | Bulk analysis of multiple texts |
| **Dashboard** | Historical stats and trends |
| **Intelligence** | Social monitoring + threat reports |
| **Fact Check** | Claim extraction and verification |
| **Education** | Learn about persuasion techniques |

### Deploy to HuggingFace Spaces (GPU)

```bash
# Prepare Space-ready folder with GPU metadata
python deploy/huggingface_space.py --prepare --gpu --output deploy/hf_space

# Push to HuggingFace Hub
python deploy/huggingface_space.py --push --repo-id YOUR_USERNAME/persuasix-studio --output deploy/hf_space
```

Manual deployment:

```bash
huggingface-cli login
huggingface-cli repo create persuasix-studio --type space --space-sdk gradio
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/persuasix-studio
git push hf main
```

---

## API Reference

The FastAPI backend exposes **20+ endpoints** with interactive Swagger documentation at `/docs`.

### Core Analysis
```
POST /api/v1/analyze              — Full text analysis
POST /api/v1/analyze/url          — Analyze URL content
POST /api/v1/analyze/file         — Analyze uploaded file
POST /api/v1/analyze/batch        — Batch analysis
POST /api/v1/analyze/spans        — Span-level detection
POST /api/v1/analyze/factcheck    — Fact-check claims
```

### Social Monitoring
```
GET  /api/v1/social/status              — Platform availability
POST /api/v1/social/mastodon/search     — Search Mastodon
POST /api/v1/social/mastodon/stream     — Real-time stream
GET  /api/v1/social/rss/fetch           — Fetch RSS feeds
POST /api/v1/social/youtube             — Video transcript analysis
POST /api/v1/social/multi-search        — Cross-platform search
```

### Intelligence
```
GET /api/v1/intelligence/active-learning     — Review candidates
GET /api/v1/intelligence/drift               — Distribution drift
GET /api/v1/intelligence/narrative-clusters   — Campaign detection
GET /api/v1/intelligence/threat-report       — Threat assessment
```

### Speech
```
POST /api/v1/speech/file    — Analyze audio/video file
POST /api/v1/speech/url     — Analyze audio/video URL
```

### Annotation
```
POST /api/v1/annotate/tasks      — Create task
GET  /api/v1/annotate/tasks/next — Next task for annotator
POST /api/v1/annotate/submit     — Submit annotation
GET  /api/v1/annotate/stats      — Statistics
POST /api/v1/annotate/export     — Export for training
```

### Reports
```
POST /api/v1/reports/pdf    — Generate PDF report
```

---

## Results

### Detection Performance (Benchmark)

| Metric | RoBERTa (ours) | TF-IDF + LR | DistilBERT |
|---|---|---|---|
| F1 Macro | **0.78** | 0.52 | 0.71 |
| F1 Micro | **0.82** | 0.58 | 0.76 |
| Exact Match | **0.45** | 0.18 | 0.35 |

### Generation Quality

| Metric | Explainer | Neutralizer |
|---|---|---|
| ROUGE-L | 0.62 | 0.55 |
| BLEU | 0.38 | 0.31 |
| BERTScore F1 | 0.87 | 0.84 |

### Cross-lingual Consistency

| Language Pair | Similarity |
|---|---|
| EN ↔ FR | 0.89 |
| EN ↔ AR | 0.82 |
| EN ↔ ES | 0.87 |
| EN ↔ DE | 0.88 |
| FR ↔ AR | 0.80 |

### Distilled Model Performance

| Model | Parameters | F1 Macro | Latency (CPU) | Size |
|-------|-----------|----------|---------------|------|
| RoBERTa-base (teacher) | 125M | 0.78 | 45ms | 500MB |
| DistilBERT (student) | 66M | 0.73 | 15ms | 130MB |
| ONNX Quantized | 66M | 0.72 | 8ms | 35MB |

---

## Deployment

### Docker

```bash
docker build -t persuasix -f docker/Dockerfile .
docker run -p 7860:7860 -p 8000:8000 persuasix
```

### Docker Compose (Full Stack)

```bash
docker-compose up -d
# Gradio UI: http://localhost:7860
# API: http://localhost:8000/docs
```

### ONNX Export

```bash
python scripts/export_model.py \
    --detector-path checkpoints/detector/final_model.pt \
    --format onnx \
    --output-dir models/exported
```

### HuggingFace Spaces (GPU)

```bash
python deploy/huggingface_space.py --prepare --gpu
python deploy/huggingface_space.py --push --repo-id YOUR_USERNAME/persuasix-studio
```

### Edge Deployment

```bash
python -m src.pipeline.distiller \
    --teacher roberta-base \
    --student distilbert-base-uncased \
    --epochs 10 \
    --export-onnx \
    --quantize dynamic
```

---

## Roadmap

- [ ] WebSocket-based live annotation presence and reviewer assignment locking
- [ ] Scheduled social monitoring jobs with persisted alerts
- [ ] Browser-side ONNX inference for the Chrome extension (fully offline)
- [ ] Graph-based campaign attribution across social platforms
- [ ] Bluesky AT Protocol integration (pending public search API)
- [ ] Telegram public channel monitoring
- [ ] Multi-modal analysis (image text extraction + meme detection)

---

## References

1. Da San Martino et al. (2020). *SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles*
2. Piskorski et al. (2023). *SemEval-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News*
3. Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*
4. Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
5. Reimers & Gurevych (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*
6. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*
7. Radford et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>PersuasiX</strong> — Because understanding manipulation is the first step to resisting it.
</p>
