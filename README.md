# PersuasiX

### Multilingual Detection, Explanation & Neutralization of Persuasion Techniques in Text

> **PersuasiX** is a large-scale multilingual NLP project that goes beyond detecting propaganda — it **explains** why a text is manipulative, **rewrites** it neutrally, and **scores** its severity across 7 languages (English, French, Arabic, Spanish, German, Chinese, Hindi).

---

## Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
- [Architecture](#architecture)
- [The 18 Persuasion Techniques](#the-18-persuasion-techniques)
- [Dataset Pipeline](#dataset-pipeline)
- [The 4 NLP Tasks](#the-4-nlp-tasks)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo App](#demo-app)
- [Results](#results)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

Misinformation and manipulative rhetoric are pervasive in modern media — from news articles and political speeches to social media posts. While existing NLP systems can *detect* propaganda, they fail at the crucial next steps: **explaining** the manipulation mechanism and **providing a neutral alternative**.

**PersuasiX** bridges this gap with a 4-stage pipeline:

```
Input Text → Detection → Explanation → Neutralization → Severity Score
```

| Feature | PersuasiX |
|---|---|
| **Languages** | English, French, Arabic, Spanish, German, Chinese, Hindi |
| **Techniques** | 18 fine-grained propaganda categories |
| **Tasks** | Detection, Explanation, Neutralization, Scoring, Fact-Checking, Span Detection |
| **Training** | LoRA/QLoRA fine-tuning for efficient adaptation |
| **Pipeline** | End-to-end: raw text → full analysis |
| **API** | FastAPI REST API with 15+ endpoints + Swagger docs |
| **Extension** | Chrome browser extension for real-time analysis |
| **Monitoring** | RSS feed monitoring with automated alerts |
| **Reports** | Professional PDF report generation |
| **Demo** | Interactive 9-tab Gradio app |

---

## Key Innovation

Most existing work stops at **detection** (binary or multi-label classification). PersuasiX introduces three novel capabilities:

1. **Explanation Generation** — Natural-language explanations of *why* each technique is manipulative, in 7 languages
2. **Text Neutralization** — Automatic rewriting that removes manipulation while preserving factual content
3. **Cross-lingual Severity Scoring** — Quantifying manipulation intensity using multilingual embeddings
4. **Span-level Detection** — Character-precise identification of manipulative phrases using RoBERTa+CRF hybrid with LLM validation
5. **Fact-Checking Integration** — Automatic claim extraction and verification via Google Fact Check Tools, ClaimBuster, and LLM-powered analysis
6. **LoRA/QLoRA Fine-tuning** — Memory-efficient training with 4-bit quantization, training only 0.5-2% of parameters
7. **Adversarial Robustness** — Tested against 12+ attack types (homoglyphs, prompt injection, semantic perturbations)
8. **Browser Extension** — Real-time Chrome extension with in-page highlighting and context menu integration
9. **Synthetic Data Pipeline** — Scalable to 200K+ examples across 7 languages using template + LLM generation

This makes PersuasiX not just a classifier, but a **complete media literacy and research platform**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PersuasiX Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │  Input Text  │──▶│  RoBERTa    │──▶│  Multi-label │             │
│  │  (EN/FR/AR)  │   │  Encoder     │   │  Detection   │             │
│  └──────────────┘   └──────────────┘   └──────┬───────┘             │
│                                                 │                   │
│                    ┌────────────────────────────┼────────────┐      │
│                    │                            │            │      │
│                    ▼                            ▼            ▼      │
│           ┌──────────────┐            ┌──────────────┐ ┌─────────┐  │
│           │  FLAN-T5     │            │  FLAN-T5     │ │Sentence │  │
│           │  Explainer   │            │  Neutralizer │ │Transf.  │  │
│           │              │            │              │ │Scorer   │  │
│           └──────┬───────┘            └──────┬───────┘ └────┬────┘  │
│                  │                           │              │       │
│                  ▼                           ▼              ▼       │
│         ┌────────────────┐          ┌────────────────┐ ┌────────┐   │
│         │ Multilingual   │          │ Neutral        │ │Severity│   │
│         │ Explanations   │          │ Rewrite        │ │ Score  │   │
│         │ (EN, FR, AR)   │          │                │ │ (0-5)  │   │
│         └────────────────┘          └────────────────┘ └────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
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

The dataset is built through a **4-stage reproducible pipeline**, inspired by the IdiomX methodology:

```
Stage 1: Collection       →  Gather texts from multiple sources
Stage 2: Cleaning         →  Deduplicate, normalize, filter
Stage 3: LLM Enrichment   →  Generate explanations + neutral rewrites via GPT-4
Stage 4: Validation       →  Semantic similarity checks + quality scoring
```

### Stage Details

| Stage | Module | What it does |
|---|---|---|
| **Collection** | `src/data/collector.py` | Aggregates texts from SemEval propaganda corpora, news articles, social media, and synthetic generation |
| **Cleaning** | `src/data/cleaner.py` | Unicode normalization, URL removal, deduplication (MD5), length filtering |
| **Enrichment** | `src/data/enricher.py` | Uses GPT-4o-mini to generate: explanations (EN/FR/AR), neutral rewrites, severity scores, target audience |
| **Validation** | `src/data/validator.py` | Validates technique labels, checks explanation quality, computes semantic similarity between original and rewrite |

### Dataset Statistics (Target)

| Metric | Value |
|---|---|
| Total examples | ~100,000+ |
| Languages | 3 (EN, FR, AR) |
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

### Task 2: Explanation Generation
- **Model**: FLAN-T5-base (fine-tuned)
- **Type**: Seq2Seq generation
- **Input**: Text + detected techniques
- **Output**: Natural-language explanation (in 3 languages)
- **Metrics**: ROUGE-L, BERTScore, BLEU

### Task 3: Text Neutralization
- **Model**: FLAN-T5-base (fine-tuned)
- **Type**: Seq2Seq generation
- **Input**: Persuasive text + techniques to remove
- **Output**: Neutral, factual rewrite
- **Metrics**: ROUGE-L, semantic preservation score

### Task 4: Cross-lingual Severity Scoring
- **Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Type**: Embedding similarity
- **Input**: Original + neutral text pair
- **Output**: Manipulation score (0-1) + severity level (0-5)

---

## Models

| Component | Base Model | Parameters | Purpose |
|---|---|---|---|
| **Detector** | `roberta-base` | 125M | Multi-label technique classification |
| **Explainer** | `google/flan-t5-base` | 250M | Explanation generation |
| **Neutralizer** | `google/flan-t5-base` | 250M | Text debiasing / rewriting |
| **Scorer** | `paraphrase-multilingual-MiniLM-L12-v2` | 118M | Cross-lingual similarity |

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
│   │   └── fact_checker.py            # Multi-API fact verification
│   └── utils/
│       ├── metrics.py                 # F1, ROUGE, BERTScore, BLEU
│       ├── visualization.py           # Plots and charts
│       └── helpers.py                 # Config, seed, device utilities
├── api/
│   ├── main.py                        # FastAPI app (15+ endpoints, Swagger)
│   ├── database.py                    # SQLAlchemy ORM (4 tables)
│   ├── schemas.py                     # Pydantic request/response models
│   └── routes/
│       ├── analysis.py                # Text/URL/File/Batch/FactCheck/Spans
│       ├── monitor.py                 # RSS feed monitoring endpoints
│       └── reports.py                 # PDF report generation
├── app/
│   ├── app.py                         # Gradio 9-tab demo application
│   ├── components.py                  # 10+ HTML component builders
│   ├── scraper.py                     # URL scraping (trafilatura + BS4)
│   ├── file_parser.py                 # PDF/DOCX/TXT parsing
│   └── history.py                     # JSON-based analysis history
├── monitor/
│   └── rss_monitor.py                 # Automated RSS feed monitoring
├── reports/
│   └── pdf_generator.py               # Professional PDF report generator
├── extension/                         # Chrome Browser Extension (Manifest V3)
│   ├── manifest.json                  # Extension configuration
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

### Optional: LLM Enrichment
To use GPT-4 for dataset enrichment, set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Quick Start

### 1. Build the dataset (offline mode, no API key needed)

```bash
python scripts/build_dataset.py --skip-enrichment
```

### 2. Train all models

```bash
python scripts/train.py --task all
```

### 3. Evaluate

```bash
python scripts/evaluate.py --task all \
    --detector-path checkpoints/detector/final_model.pt \
    --explainer-path checkpoints/explainer/final_model \
    --neutralizer-path checkpoints/neutralizer/final_model
```

### 4. Launch demo

```bash
python app/app.py
# Open http://localhost:7860
```

### Using the Pipeline in Code

```python
from src.pipeline.persuasix_pipeline import PersuasixPipeline

# Load with default models (or provide checkpoint paths)
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

### Training Features
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

### Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Demo App

The Gradio-powered **PersuasiX Studio** provides:

- **Technique Detection** — Color-coded badges with confidence bars
- **Multilingual Explanation** — Explanations in EN, FR, AR
- **Severity Gauge** — Visual manipulation intensity meter
- **Side-by-side Comparison** — Original vs neutralized text
- **Raw JSON** — Full analysis output for developers

### Deploy to HuggingFace Spaces

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create Space and push
huggingface-cli repo create persuasix-studio --type space --space-sdk gradio
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/persuasix-studio
git push hf main
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
| FR ↔ AR | 0.80 |

---

## Deployment

### Docker

```bash
docker build -t persuasix -f docker/Dockerfile .
docker run -p 7860:7860 persuasix
```

### ONNX Export

```bash
python scripts/export_model.py \
    --detector-path checkpoints/detector/final_model.pt \
    --format onnx \
    --output-dir models/exported
```

---

## Implemented Advanced Features

- [x] **Scale dataset to 200K+ examples** — `scripts/generate_synthetic_data.py` with template + LLM + paraphrase augmentation across 7 languages
- [x] **Add more languages** — Spanish, German, Chinese, Hindi added (7 total)
- [x] **LoRA/QLoRA fine-tuning** — `src/training/lora_trainer.py` with 4-bit quantization, configurable rank/alpha, multi-task support
- [x] **Span-level detection** — `src/pipeline/span_detector.py` with RoBERTa+CRF model + LLM hybrid detector
- [x] **Browser extension** — `extension/` Chrome Manifest V3 extension with popup, context menu, in-page highlighting
- [x] **Fact-checking API integration** — `src/pipeline/fact_checker.py` with Google Fact Check Tools + ClaimBuster + LLM verification
- [x] **Adversarial robustness testing** — `tests/adversarial/` with 12+ attack types and automated robustness scoring

## Future Work

- [ ] Deploy to HuggingFace Spaces with GPU inference
- [ ] Add real-time collaborative annotation interface
- [ ] Integrate with social media APIs (Twitter/X, Reddit) for live monitoring
- [ ] Build model distillation pipeline for edge deployment
- [ ] Add speech-to-text for audio/video persuasion analysis
- [ ] Publish Chrome extension to Web Store

---

## References

1. Da San Martino et al. (2020). *SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles*
2. Piskorski et al. (2023). *SemEval-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News*
3. Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*
4. Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
5. Reimers & Gurevych (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>PersuasiX</strong> — Because understanding manipulation is the first step to resisting it.
</p>
