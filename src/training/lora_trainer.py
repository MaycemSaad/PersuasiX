"""
LoRA / QLoRA fine-tuning for PersuasiX models.

Supports:
  - LoRA fine-tuning (low-rank adaptation with frozen base weights)
  - QLoRA fine-tuning (4-bit quantized base + LoRA adapters)
  - Multi-task training (detection + explanation + neutralization)
  - Configurable rank, alpha, target modules
  - Gradient checkpointing for memory efficiency
  - WandB experiment tracking

Usage:
    python -m src.training.lora_trainer --task detection --base-model roberta-large --rank 16
    python -m src.training.lora_trainer --task explanation --base-model google/flan-t5-large --quantize 4bit
    python -m src.training.lora_trainer --task neutralization --base-model google/flan-t5-xl --quantize 4bit --rank 32
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not installed. Install with: pip install peft")

try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Configuration for LoRA/QLoRA fine-tuning."""

    # Task
    task: str = "detection"  # detection | explanation | neutralization | span_detection
    num_labels: int = 18  # For detection task

    # Base model
    base_model: str = "roberta-large"
    tokenizer_name: str | None = None  # defaults to base_model

    # LoRA hyperparameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None  # auto-detect if None
    bias: str = "none"  # none | all | lora_only
    modules_to_save: list[str] | None = None  # additional modules to train fully

    # Quantization
    quantize: str = "none"  # none | 4bit | 8bit
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.06
    fp16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Data
    train_data: str = "data/synthetic/train.jsonl"
    val_data: str = "data/synthetic/val.jsonl"

    # Output
    output_dir: str = "checkpoints/lora"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50

    # Tracking
    use_wandb: bool = True
    wandb_project: str = "persuasix-lora"
    wandb_run_name: str | None = None

    # Early stopping
    patience: int = 3
    min_delta: float = 0.001

    def get_peft_task_type(self) -> str:
        if self.task in ("detection", "span_detection"):
            return "SEQ_CLS"
        return "SEQ_2_SEQ_LM"

    def auto_detect_target_modules(self) -> list[str]:
        """Auto-detect LoRA target modules based on model architecture."""
        model_lower = self.base_model.lower()
        if "roberta" in model_lower or "bert" in model_lower:
            return ["query", "key", "value", "dense"]
        elif "t5" in model_lower:
            return ["q", "k", "v", "o", "wi", "wo"]
        elif "llama" in model_lower or "mistral" in model_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt" in model_lower:
            return ["c_attn", "c_proj", "c_fc"]
        else:
            return ["query", "key", "value"]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

class PersuasionDetectionDataset(Dataset):
    """Multi-label classification dataset for persuasion detection."""

    TECHNIQUE_LABELS = [
        "appeal_to_fear", "appeal_to_authority", "bandwagon", "false_dilemma",
        "ad_hominem", "straw_man", "red_herring", "loaded_language",
        "whataboutism", "causal_oversimplification", "appeal_to_emotion",
        "repetition", "exaggeration", "doubt", "slogans", "name_calling",
        "flag_waving", "thought_terminating_cliche",
    ]

    def __init__(self, data_path: str, tokenizer, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict] = []

        path = Path(data_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.examples.append(json.loads(line))
            logger.info(f"Loaded {len(self.examples)} examples from {path}")
        else:
            logger.warning(f"Data file not found: {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = ex["text"]
        techniques = ex.get("techniques", [])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Multi-label encoding
        labels = torch.zeros(len(self.TECHNIQUE_LABELS), dtype=torch.float)
        for t in techniques:
            if t in self.TECHNIQUE_LABELS:
                labels[self.TECHNIQUE_LABELS.index(t)] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


class Seq2SeqPersuasionDataset(Dataset):
    """Seq2seq dataset for explanation/neutralization tasks."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        task: str = "explanation",
        max_source_length: int = 512,
        max_target_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.task = task
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples: list[dict] = []

        path = Path(data_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ex = json.loads(line)
                        if self._has_target(ex):
                            self.examples.append(ex)
            logger.info(f"Loaded {len(self.examples)} {task} examples from {path}")

    def _has_target(self, ex: dict) -> bool:
        if self.task == "explanation":
            return bool(ex.get("explanation_en") or ex.get("explanation"))
        elif self.task == "neutralization":
            return bool(ex.get("neutral_rewrite"))
        return False

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = ex["text"]
        techniques = ex.get("techniques", [])

        if self.task == "explanation":
            source = f"Explain persuasion techniques in: {text}\nTechniques: {', '.join(techniques)}"
            target = ex.get("explanation_en", ex.get("explanation", ""))
        else:
            source = f"Neutralize the following text by removing persuasion: {text}"
            target = ex.get("neutral_rewrite", text)

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


class SpanDetectionDataset(Dataset):
    """Token-level span detection dataset."""

    TECHNIQUE_TO_ID = {t: i + 1 for i, t in enumerate([
        "appeal_to_fear", "appeal_to_authority", "bandwagon", "false_dilemma",
        "ad_hominem", "straw_man", "red_herring", "loaded_language",
        "whataboutism", "causal_oversimplification", "appeal_to_emotion",
        "repetition", "exaggeration", "doubt", "slogans", "name_calling",
        "flag_waving", "thought_terminating_cliche",
    ])}  # 0 = O (no technique)

    def __init__(self, data_path: str, tokenizer, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict] = []

        path = Path(data_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ex = json.loads(line)
                        if ex.get("spans"):
                            self.examples.append(ex)
            logger.info(f"Loaded {len(self.examples)} span examples from {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = ex["text"]
        spans = ex.get("spans", [])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping").squeeze(0)
        labels = torch.zeros(self.max_length, dtype=torch.long)

        for span in spans:
            start_char = span.get("start", 0)
            end_char = span.get("end", 0)
            technique = span.get("technique", "")
            tech_id = self.TECHNIQUE_TO_ID.get(technique, 0)

            for token_idx in range(self.max_length):
                token_start, token_end = offset_mapping[token_idx].tolist()
                if token_start == 0 and token_end == 0:
                    continue
                if token_start >= start_char and token_end <= end_char:
                    labels[token_idx] = tech_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# LoRA Trainer
# ---------------------------------------------------------------------------

class LoRATrainer:
    """
    LoRA/QLoRA fine-tuning trainer for PersuasiX models.

    Supports:
    - Detection: multi-label classification with LoRA on encoder
    - Explanation: seq2seq generation with LoRA on encoder-decoder
    - Neutralization: seq2seq generation with LoRA
    - Span Detection: token classification with LoRA
    """

    def __init__(self, config: LoRAConfig) -> None:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required. Install: pip install peft")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required. Install: pip install transformers")

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"LoRA Trainer initialized: task={config.task}, model={config.base_model}")
        logger.info(f"  Rank={config.lora_rank}, Alpha={config.lora_alpha}, Quantize={config.quantize}")
        logger.info(f"  Device={self.device}, FP16={config.fp16}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._build_model()
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(self.train_loader) * config.epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)

        from torch.optim.lr_scheduler import OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
        )

        self.scaler = torch.amp.GradScaler("cuda") if config.fp16 and self.device == "cuda" else None
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self._print_trainable_params()

    def _load_tokenizer(self) -> Any:
        name = self.config.tokenizer_name or self.config.base_model
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_quantization_config(self) -> Any:
        if self.config.quantize == "4bit":
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype, torch.float16)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.use_double_quant,
            )
        elif self.config.quantize == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _build_model(self) -> nn.Module:
        """Build base model + LoRA adapters."""
        quant_config = self._get_quantization_config()
        model_kwargs: dict[str, Any] = {}
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"

        # Load base model
        if self.config.task == "detection":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.base_model,
                num_labels=self.config.num_labels,
                problem_type="multi_label_classification",
                **model_kwargs,
            )
        elif self.config.task == "span_detection":
            num_span_labels = len(SpanDetectionDataset.TECHNIQUE_TO_ID) + 1  # +1 for O tag
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.base_model,
                num_labels=num_span_labels,
                **model_kwargs,
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.base_model,
                **model_kwargs,
            )

        # Prepare for k-bit training if quantized
        if quant_config:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )
        elif self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Configure LoRA
        target_modules = self.config.target_modules or self.config.auto_detect_target_modules()

        peft_task = TaskType.SEQ_CLS if self.config.task in ("detection", "span_detection") else TaskType.SEQ_2_SEQ_LM

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias=self.config.bias,
            task_type=peft_task,
            modules_to_save=self.config.modules_to_save,
        )

        model = get_peft_model(model, lora_config)

        if not quant_config:
            model = model.to(self.device)

        return model

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation dataloaders."""
        if self.config.task == "detection":
            train_dataset = PersuasionDetectionDataset(
                self.config.train_data, self.tokenizer, self.config.max_seq_length,
            )
            val_dataset = PersuasionDetectionDataset(
                self.config.val_data, self.tokenizer, self.config.max_seq_length,
            )
        elif self.config.task == "span_detection":
            train_dataset = SpanDetectionDataset(
                self.config.train_data, self.tokenizer, self.config.max_seq_length,
            )
            val_dataset = SpanDetectionDataset(
                self.config.val_data, self.tokenizer, self.config.max_seq_length,
            )
        else:
            train_dataset = Seq2SeqPersuasionDataset(
                self.config.train_data, self.tokenizer, self.config.task,
                self.config.max_seq_length,
            )
            val_dataset = Seq2SeqPersuasionDataset(
                self.config.val_data, self.tokenizer, self.config.task,
                self.config.max_seq_length,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _print_trainable_params(self) -> None:
        """Print trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = 100 * trainable / max(total, 1)
        logger.info(
            f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%) "
            f"| Memory saved: ~{(1 - pct / 100) * 100:.0f}%"
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list[float]]:
        """Run the full LoRA fine-tuning loop."""
        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "learning_rate": [], "epoch_time": [],
        }

        # Initialize WandB
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or f"lora-{self.config.task}-r{self.config.lora_rank}",
                config={
                    "task": self.config.task,
                    "base_model": self.config.base_model,
                    "lora_rank": self.config.lora_rank,
                    "lora_alpha": self.config.lora_alpha,
                    "quantize": self.config.quantize,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "epochs": self.config.epochs,
                },
            )

        logger.info(f"Starting LoRA training: {self.config.epochs} epochs")

        for epoch in range(1, self.config.epochs + 1):
            start = time.time()

            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            elapsed = time.time() - start

            lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(lr)
            history["epoch_time"].append(elapsed)

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                f"lr: {lr:.2e} | time: {elapsed:.1f}s"
            )

            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "lr": lr, "epoch_time": elapsed,
                })

            # Checkpointing
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_steps = 0

        progress = tqdm(self.train_loader, desc=f"Train epoch {epoch}", leave=False)
        self.optimizer.zero_grad()

        for step, batch in enumerate(progress):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

            total_loss += outputs.loss.item()
            num_steps += 1

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm,
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_steps = 0

        for batch in tqdm(self.val_loader, desc=f"Val epoch {epoch}", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_steps += 1

        return total_loss / max(num_steps, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save LoRA adapter weights (NOT full model)."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save only adapter weights
        adapter_dir = output_dir / f"adapter_epoch{epoch}"
        self.model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))

        if is_best:
            best_dir = output_dir / "best_adapter"
            self.model.save_pretrained(str(best_dir))
            self.tokenizer.save_pretrained(str(best_dir))
            logger.info(f"Best adapter saved: val_loss={val_loss:.4f} -> {best_dir}")

        # Save training config
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "task": self.config.task,
                "base_model": self.config.base_model,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "quantize": self.config.quantize,
                "best_val_loss": self.best_val_loss,
                "epoch": epoch,
            }, f, indent=2)

    # ------------------------------------------------------------------
    # Inference with LoRA
    # ------------------------------------------------------------------

    @classmethod
    def load_for_inference(
        cls,
        adapter_path: str,
        base_model: str | None = None,
        device: str = "auto",
    ) -> tuple[nn.Module, Any]:
        """Load a trained LoRA adapter for inference."""
        adapter_path = Path(adapter_path)
        config_path = adapter_path.parent / "training_config.json"

        if config_path.exists():
            with open(config_path) as f:
                training_config = json.load(f)
            base_model = base_model or training_config["base_model"]
            task = training_config["task"]
        else:
            task = "detection"

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

        if task == "detection":
            base = AutoModelForSequenceClassification.from_pretrained(base_model)
        else:
            base = AutoModelForSeq2SeqLM.from_pretrained(base_model)

        model = PeftModel.from_pretrained(base, str(adapter_path))
        model = model.to(device)
        model.eval()

        logger.info(f"Loaded LoRA adapter from {adapter_path} (base: {base_model})")
        return model, tokenizer

    @classmethod
    def merge_and_export(
        cls,
        adapter_path: str,
        output_path: str,
        base_model: str | None = None,
    ) -> None:
        """Merge LoRA weights into base model and export as a standalone model."""
        model, tokenizer = cls.load_for_inference(adapter_path, base_model)
        merged = model.merge_and_unload()
        merged.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Merged model exported to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for PersuasiX")
    parser.add_argument("--task", choices=["detection", "explanation", "neutralization", "span_detection"], default="detection")
    parser.add_argument("--base-model", type=str, default="roberta-large")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--quantize", choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-data", type=str, default="data/synthetic/train.jsonl")
    parser.add_argument("--val-data", type=str, default="data/synthetic/val.jsonl")
    parser.add_argument("--output-dir", type=str, default="checkpoints/lora")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = LoRAConfig(
        task=args.task,
        base_model=args.base_model,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        quantize=args.quantize,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )

    trainer = LoRATrainer(config)
    history = trainer.train()

    logger.success(f"Training complete! Best val_loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Adapter saved to: {config.output_dir}/best_adapter/")


if __name__ == "__main__":
    main()
