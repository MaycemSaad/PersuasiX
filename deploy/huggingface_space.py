#!/usr/bin/env python3
"""
HuggingFace Spaces deployment helper for PersuasiX.

Generates the required files for deploying PersuasiX as a HuggingFace Space
with optional GPU inference (ZeroGPU / T4 / A10G).

Usage:
    python deploy/huggingface_space.py --prepare          # Generate Space files
    python deploy/huggingface_space.py --push             # Push to HF Spaces
    python deploy/huggingface_space.py --prepare --gpu    # GPU-enabled Space
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger


SPACE_README = """\
---
title: PersuasiX
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
tags:
  - nlp
  - persuasion-detection
  - propaganda
  - multilingual
  - media-literacy
  - fact-checking
{gpu_line}
---

# PersuasiX — Multilingual Persuasion Detection Platform

Detect, explain, and neutralize persuasion techniques in text across 7 languages.

**Features:** 18 techniques | 7 languages | Span detection | Fact checking | PDF reports

🔗 [GitHub Repository](https://github.com/MaycemSaad/PersuasiX)
"""

SPACE_APP = """\
#!/usr/bin/env python3
\"\"\"PersuasiX — HuggingFace Spaces Entry Point.\"\"\"

import os
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# GPU setup for ZeroGPU
DEVICE = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"GPU available: {{torch.cuda.get_device_name(0)}}")
except ImportError:
    pass

# Import and launch the app
from app.app import create_demo
from app.components import CSS

demo = create_demo()
demo.queue(max_size=20)
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    css=CSS,
    show_error=True,
    max_threads=4,
)
"""

SPACE_REQUIREMENTS = """\
# Core
torch>=2.1.0
transformers>=4.36.0
sentence-transformers>=2.3.0
accelerate>=0.25.0
peft>=0.7.0

# LLM
openai>=1.6.0
tiktoken>=0.5.0

# Web scraping
requests>=2.31.0
beautifulsoup4>=4.12.0
trafilatura>=1.6.0

# Document parsing
pypdf>=4.0.0
python-docx>=1.1.0

# API
fastapi>=0.110.0
uvicorn>=0.27.0
sqlalchemy>=2.0.0

# Monitoring
feedparser>=6.0.0
apscheduler>=3.10.0

# Reports
fpdf2>=2.7.0

# App
gradio>=4.12.0

# Utilities
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
loguru>=0.7.0
tqdm>=4.66.0
"""

# Files to include in the Space
INCLUDE_DIRS = [
    "src/", "app/", "api/", "monitor/", "reports/",
    "config/", "data/sample/", "data/raw/", "data/processed/",
]

INCLUDE_FILES = [
    "setup.py", "requirements.txt",
]

EXCLUDE_PATTERNS = [
    "__pycache__", ".pyc", ".git", ".env", ".history",
    "*.db", "*.pdf", "node_modules", "checkpoints/",
    "data/synthetic/", "data/adversarial_report.json",
    "extension/", "tests/", "notebooks/", "docker/",
]


def prepare_space(output_dir: Path, gpu: bool = False) -> None:
    """Generate all files needed for HuggingFace Spaces deployment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent

    logger.info(f"Preparing HuggingFace Space in {output_dir}")

    # 1. Write Space README
    gpu_line = "hardware: t4-small" if gpu else ""
    readme = SPACE_README.format(gpu_line=gpu_line)
    (output_dir / "README.md").write_text(readme)
    logger.info("  Created README.md (Space metadata)")

    # 2. Write Space app entry point
    (output_dir / "app.py").write_text(SPACE_APP)
    logger.info("  Created app.py (entry point)")

    # 3. Write requirements
    (output_dir / "requirements.txt").write_text(SPACE_REQUIREMENTS.strip())
    logger.info("  Created requirements.txt")

    # 4. Copy project files
    for dir_path in INCLUDE_DIRS:
        src = project_root / dir_path
        dst = output_dir / dir_path
        if src.exists():
            _copy_dir(src, dst)
            logger.info(f"  Copied {dir_path}")

    for file_path in INCLUDE_FILES:
        src = project_root / file_path
        dst = output_dir / file_path
        if src.exists():
            shutil.copy2(str(src), str(dst))

    # 5. Create .gitkeep files for empty dirs
    for d in ["data/raw", "data/processed", "data/reports"]:
        p = output_dir / d
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()

    # 6. Write startup script
    startup = output_dir / "startup.sh"
    startup.write_text("""\
#!/bin/bash
pip install -e . 2>/dev/null || true
python app.py
""")
    startup.chmod(0o755)

    total_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
    logger.success(f"Space prepared: {total_files} files in {output_dir}")


def push_to_hf(space_dir: Path, repo_id: str) -> None:
    """Push the prepared Space to HuggingFace."""
    logger.info(f"Pushing to HuggingFace: {repo_id}")

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )

        api.upload_folder(
            folder_path=str(space_dir),
            repo_id=repo_id,
            repo_type="space",
        )

        logger.success(f"Deployed to: https://huggingface.co/spaces/{repo_id}")

    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        logger.info("Alternative: use git to push manually:")
        logger.info(f"  cd {space_dir}")
        logger.info(f"  git init && git add .")
        logger.info(f'  git commit -m "Deploy PersuasiX"')
        logger.info(f"  git remote add hf https://huggingface.co/spaces/{repo_id}")
        logger.info(f"  git push hf main")


def _copy_dir(src: Path, dst: Path) -> None:
    """Copy directory excluding unwanted patterns."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if any(pat in str(item) for pat in EXCLUDE_PATTERNS):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(target))


def main():
    parser = argparse.ArgumentParser(description="Deploy PersuasiX to HuggingFace Spaces")
    parser.add_argument("--prepare", action="store_true", help="Prepare Space files")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (T4)")
    parser.add_argument("--output", type=str, default="deploy/hf_space", help="Output directory")
    parser.add_argument("--repo-id", type=str, default="MaycemSaad/PersuasiX", help="HF repo ID")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.prepare:
        prepare_space(output_dir, gpu=args.gpu)

    if args.push:
        push_to_hf(output_dir, args.repo_id)

    if not args.prepare and not args.push:
        parser.print_help()


if __name__ == "__main__":
    main()
