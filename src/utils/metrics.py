"""Custom evaluation metrics for PersuasiX."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    hamming_loss,
)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict:
    """Compute comprehensive multi-label classification metrics."""
    metrics = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float(np.all(y_pred == y_true, axis=1).mean()),
    }

    per_technique = {}
    for i, name in enumerate(label_names):
        if y_true[:, i].sum() > 0:
            per_technique[name] = {
                "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "support": int(y_true[:, i].sum()),
            }
    metrics["per_technique"] = per_technique

    return metrics


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute ROUGE, BLEU, and BERTScore for generated text."""
    metrics: dict = {}

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        for key in rouge_scores:
            metrics[key] = float(np.mean(rouge_scores[key]))
    except ImportError:
        pass

    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        metrics["bleu"] = bleu.score / 100
    except ImportError:
        pass

    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            predictions, references,
            lang="en",
            verbose=False,
        )
        metrics["bertscore_precision"] = float(P.mean())
        metrics["bertscore_recall"] = float(R.mean())
        metrics["bertscore_f1"] = float(F1.mean())
    except ImportError:
        pass

    return metrics
