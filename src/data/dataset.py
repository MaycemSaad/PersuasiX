"""PyTorch Dataset classes for all PersuasiX tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .collector import TECHNIQUE_LABELS


class PersuasixDataset(Dataset):
    """Multi-label classification dataset for persuasion technique detection."""

    def __init__(
        self,
        data: pd.DataFrame | str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label_columns: list[str] | None = None,
    ) -> None:
        if isinstance(data, (str, Path)):
            data = pd.read_json(data, lines=True)
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = label_columns or TECHNIQUE_LABELS

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        text = str(row["text"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self._encode_labels(row.get("techniques", []))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def _encode_labels(self, techniques: list[str]) -> torch.Tensor:
        label_vec = torch.zeros(len(self.label_names), dtype=torch.float)
        if isinstance(techniques, list):
            for tech in techniques:
                if tech in self.label_names:
                    label_vec[self.label_names.index(tech)] = 1.0
        return label_vec

    @staticmethod
    def decode_labels(predictions: torch.Tensor, threshold: float = 0.5) -> list[list[str]]:
        results = []
        probs = torch.sigmoid(predictions)
        for row in probs:
            techniques = [
                TECHNIQUE_LABELS[i]
                for i, val in enumerate(row)
                if val >= threshold
            ]
            results.append(techniques)
        return results


class ExplainerDataset(Dataset):
    """Seq2seq dataset: text + techniques → explanation."""

    def __init__(
        self,
        data: pd.DataFrame | str | Path,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 256,
        target_language: str = "en",
    ) -> None:
        if isinstance(data, (str, Path)):
            data = pd.read_json(data, lines=True)

        explanation_col = f"explanation_{target_language}"
        self.data = data[data[explanation_col].notna()].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.target_language = target_language
        self.explanation_col = explanation_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        techniques = row.get("techniques", [])
        tech_str = ", ".join(t.replace("_", " ") for t in techniques) if techniques else "none"

        input_text = (
            f"Explain the persuasion techniques in this text.\n"
            f"Techniques: {tech_str}\n"
            f"Text: {row['text']}"
        )
        target_text = str(row[self.explanation_col])

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


class NeutralizerDataset(Dataset):
    """Seq2seq dataset: persuasive text → neutral rewrite."""

    def __init__(
        self,
        data: pd.DataFrame | str | Path,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 512,
    ) -> None:
        if isinstance(data, (str, Path)):
            data = pd.read_json(data, lines=True)

        has_rewrite = (
            data["neutral_rewrite"].notna()
            & (data["neutral_rewrite"] != "[Neutral rewrite requires LLM enrichment]")
            & (data["is_persuasive"] == True)  # noqa: E712
        )
        self.data = data[has_rewrite].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        techniques = row.get("techniques", [])
        tech_str = ", ".join(t.replace("_", " ") for t in techniques) if techniques else "none"

        input_text = (
            f"Neutralize the following persuasive text by removing manipulation techniques.\n"
            f"Techniques to remove: {tech_str}\n"
            f"Text: {row['text']}"
        )
        target_text = str(row["neutral_rewrite"])

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
