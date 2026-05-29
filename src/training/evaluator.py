"""Evaluate all PersuasiX tasks with comprehensive metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from ..data.collector import TECHNIQUE_LABELS
from ..utils.metrics import compute_detection_metrics, compute_generation_metrics


class PersuasixEvaluator:
    """Unified evaluator for detection, explanation, and neutralization."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Task 1: Detection evaluation
    # ------------------------------------------------------------------

    def evaluate_detector(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        threshold: float = 0.5,
    ) -> dict:
        """Evaluate multi-label detection with per-technique and aggregate metrics."""
        model.eval()
        model.to(self.device)

        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Evaluating detector"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs["logits"]).cpu()

            preds = (probs >= threshold).long()
            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        metrics = compute_detection_metrics(all_labels, all_preds, TECHNIQUE_LABELS)
        logger.info(f"Detection — F1 macro: {metrics['f1_macro']:.4f} | F1 micro: {metrics['f1_micro']:.4f}")
        return metrics

    # ------------------------------------------------------------------
    # Task 2 & 3: Generation evaluation
    # ------------------------------------------------------------------

    def evaluate_generator(
        self,
        model,
        tokenizer,
        dataloader: DataLoader,
        task_name: str = "explainer",
        max_new_tokens: int = 256,
    ) -> dict:
        """Evaluate seq2seq generation with ROUGE, BERTScore, and BLEU."""
        if hasattr(model, "model"):
            gen_model = model.model
        else:
            gen_model = model

        gen_model.eval()
        gen_model.to(self.device)

        all_predictions = []
        all_references = []

        for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            with torch.no_grad():
                generated = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                )

            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_predictions.extend(pred_texts)

            label_ids = labels.clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            all_references.extend(ref_texts)

        metrics = compute_generation_metrics(all_predictions, all_references)
        logger.info(
            f"{task_name} — ROUGE-L: {metrics.get('rougeL', 0):.4f} | "
            f"BLEU: {metrics.get('bleu', 0):.4f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Full evaluation report
    # ------------------------------------------------------------------

    def full_report(self, results: dict[str, dict]) -> str:
        """Format a complete evaluation report."""
        lines = ["=" * 60, "  PersuasiX — Evaluation Report", "=" * 60, ""]

        for task, metrics in results.items():
            lines.append(f"--- {task.upper()} ---")
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key:30s}: {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, float):
                            lines.append(f"    {sub_key:28s}: {sub_val:.4f}")
                else:
                    lines.append(f"  {key:30s}: {value}")
            lines.append("")

        report = "\n".join(lines)
        logger.info(f"\n{report}")
        return report
