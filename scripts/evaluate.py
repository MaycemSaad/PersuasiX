"""CLI — Evaluate trained PersuasiX models on the test set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.models.detector import PersuasionDetector
from src.data.dataset import PersuasixDataset, ExplainerDataset, NeutralizerDataset
from src.training.evaluator import PersuasixEvaluator
from src.utils.helpers import load_config, set_seed, get_device


@click.command()
@click.option("--config", default="config/config.yaml")
@click.option("--task", type=click.Choice(["detector", "explainer", "neutralizer", "all"]), default="all")
@click.option("--detector-path", default=None, help="Path to detector checkpoint")
@click.option("--explainer-path", default=None, help="Path to explainer checkpoint")
@click.option("--neutralizer-path", default=None, help="Path to neutralizer checkpoint")
def main(config: str, task: str, detector_path: str, explainer_path: str, neutralizer_path: str):
    """Evaluate PersuasiX models."""
    cfg = load_config(config)
    set_seed(cfg["project"]["seed"])
    device = get_device()

    data_dir = Path(cfg["data"]["processed_dir"])
    evaluator = PersuasixEvaluator(device=device)
    results = {}

    if task in ("detector", "all") and detector_path:
        logger.info("=== Evaluating Detector ===")
        model = PersuasionDetector.from_pretrained(detector_path)
        tokenizer = PersuasionDetector.get_tokenizer(cfg["detector"]["model_name"])
        test_ds = PersuasixDataset(data_dir / "test.json", tokenizer)
        test_loader = DataLoader(test_ds, batch_size=cfg["detector"]["batch_size"])
        results["detector"] = evaluator.evaluate_detector(model, test_loader)

    if task in ("explainer", "all") and explainer_path:
        logger.info("=== Evaluating Explainer ===")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(explainer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(explainer_path)
        test_ds = ExplainerDataset(data_dir / "test.json", tokenizer)
        test_loader = DataLoader(test_ds, batch_size=cfg["explainer"]["batch_size"])
        results["explainer"] = evaluator.evaluate_generator(model, tokenizer, test_loader, "explainer")

    if task in ("neutralizer", "all") and neutralizer_path:
        logger.info("=== Evaluating Neutralizer ===")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(neutralizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(neutralizer_path)
        test_ds = NeutralizerDataset(data_dir / "test.json", tokenizer)
        test_loader = DataLoader(test_ds, batch_size=cfg["neutralizer"]["batch_size"])
        results["neutralizer"] = evaluator.evaluate_generator(model, tokenizer, test_loader, "neutralizer")

    if results:
        report = evaluator.full_report(results)
        report_path = Path(cfg["training"]["logging_dir"]) / "evaluation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        logger.success(f"Report saved: {report_path}")
    else:
        logger.warning("No models evaluated. Provide checkpoint paths with --detector-path, etc.")


if __name__ == "__main__":
    main()
