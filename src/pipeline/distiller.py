"""
Knowledge distillation pipeline for edge deployment.

Compresses the large PersuasiX models into smaller, faster models
suitable for mobile, browser (ONNX.js), and embedded deployment.

Techniques:
  1. Knowledge Distillation (teacher → student)
  2. Model Pruning (structured + unstructured)
  3. Quantization (INT8, FP16, dynamic)
  4. ONNX export with optimization
  5. TFLite conversion for mobile

Architecture:
  Teacher: RoBERTa-large (355M params)
  Student: TinyBERT/DistilBERT/MobileBERT (14-67M params)

Usage:
    python -m src.pipeline.distiller --teacher roberta-large --student distilbert-base-uncased
    python -m src.pipeline.distiller --export-onnx --quantize int8
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    # Teacher model
    teacher_model: str = "roberta-large"
    teacher_checkpoint: str | None = None  # Path to fine-tuned teacher

    # Student model
    student_model: str = "distilbert-base-uncased"
    num_labels: int = 18

    # Distillation hyperparameters
    temperature: float = 4.0           # Softmax temperature for soft targets
    alpha_ce: float = 0.5              # Weight for hard label loss
    alpha_kd: float = 0.5              # Weight for distillation loss
    alpha_hidden: float = 0.1          # Weight for hidden state matching
    alpha_attention: float = 0.0       # Weight for attention transfer

    # Training
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 256          # Shorter for edge deployment
    gradient_accumulation_steps: int = 2

    # Pruning
    pruning_amount: float = 0.3        # Fraction of weights to prune
    structured_pruning: bool = False

    # Quantization
    quantize: str = "none"             # none, dynamic, static, int8

    # Export
    export_onnx: bool = True
    export_tflite: bool = False
    onnx_opset: int = 14
    optimize_onnx: bool = True

    # Data
    train_data: str = "data/synthetic/train.jsonl"
    val_data: str = "data/synthetic/val.jsonl"
    output_dir: str = "models/distilled"


# ---------------------------------------------------------------------------
# Distillation Loss
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
      L = alpha_ce * CE(student, labels) + alpha_kd * KL(student_soft, teacher_soft)
          + alpha_hidden * MSE(student_hidden, teacher_hidden)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha_ce: float = 0.5,
        alpha_kd: float = 0.5,
        alpha_hidden: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.alpha_hidden = alpha_hidden
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: torch.Tensor | None = None,
        teacher_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        # Hard label loss (cross-entropy)
        ce = self.ce_loss(student_logits, labels)

        # Soft label loss (KL divergence with temperature scaling)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        kd = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.temperature ** 2)

        total = self.alpha_ce * ce + self.alpha_kd * kd

        losses = {"total": total, "ce": ce, "kd": kd}

        # Hidden state matching (optional)
        if self.alpha_hidden > 0 and student_hidden is not None and teacher_hidden is not None:
            if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
                # Project student hidden states to teacher dimension
                proj = nn.Linear(student_hidden.shape[-1], teacher_hidden.shape[-1]).to(student_hidden.device)
                student_hidden = proj(student_hidden)
            hidden_loss = F.mse_loss(student_hidden, teacher_hidden)
            total = total + self.alpha_hidden * hidden_loss
            losses["hidden"] = hidden_loss
            losses["total"] = total

        return losses


# ---------------------------------------------------------------------------
# Student Model Wrapper
# ---------------------------------------------------------------------------

class StudentClassifier(nn.Module):
    """Lightweight student classifier for distillation."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 18,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        result = {"logits": logits}
        if output_hidden:
            result["hidden_states"] = cls_output
        return result


# ---------------------------------------------------------------------------
# Distillation Trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """Train a student model using knowledge distillation from a teacher."""

    def __init__(self, config: DistillationConfig) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required: pip install transformers")

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Distillation: {config.teacher_model} -> {config.student_model}")
        logger.info(f"  Device: {self.device} | Temperature: {config.temperature}")

        # Load teacher
        self.teacher = self._load_teacher()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Build student
        self.student = StudentClassifier(
            model_name=config.student_model,
            num_labels=config.num_labels,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Loss
        self.criterion = DistillationLoss(
            temperature=config.temperature,
            alpha_ce=config.alpha_ce,
            alpha_kd=config.alpha_kd,
            alpha_hidden=config.alpha_hidden,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
        )

        self._print_model_sizes()

    def _load_teacher(self) -> nn.Module:
        """Load the teacher model."""
        if self.config.teacher_checkpoint:
            from src.models.detector import PersuasionDetector
            model = PersuasionDetector.from_pretrained(self.config.teacher_checkpoint)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.teacher_model,
                num_labels=self.config.num_labels,
                problem_type="multi_label_classification",
            )
        return model.to(self.device)

    def _print_model_sizes(self) -> None:
        """Print and compare model sizes."""
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        compression = teacher_params / max(student_params, 1)
        logger.info(
            f"  Teacher: {teacher_params:,} params | "
            f"Student: {student_params:,} params | "
            f"Compression: {compression:.1f}x"
        )

    def train(self) -> dict:
        """Run the distillation training loop."""
        # Build a simple dataset from JSONL
        train_loader = self._build_dataloader(self.config.train_data, shuffle=True)
        val_loader = self._build_dataloader(self.config.val_data, shuffle=False)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "kd_loss": []}

        for epoch in range(1, self.config.epochs + 1):
            start = time.time()
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader, epoch)
            elapsed = time.time() - start

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["kd_loss"].append(train_metrics.get("kd", 0))

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train_loss: {train_metrics['total']:.4f} | "
                f"val_loss: {val_metrics['total']:.4f} | "
                f"kd: {train_metrics.get('kd', 0):.4f} | "
                f"time: {elapsed:.1f}s"
            )

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                self._save_student("best")

        # Final save
        self._save_student("final")

        # Export
        if self.config.export_onnx:
            self._export_onnx()

        if self.config.quantize != "none":
            self._quantize_model()

        return history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.student.train()
        total_losses = {"total": 0, "ce": 0, "kd": 0}
        n_batches = 0

        for batch in tqdm(loader, desc=f"Distill epoch {epoch}", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out = self.teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                teacher_logits = teacher_out.get("logits", teacher_out.get("logits"))

            # Student forward
            student_out = self.student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden=self.config.alpha_hidden > 0,
            )

            # Compute distillation loss
            losses = self.criterion(
                student_logits=student_out["logits"],
                teacher_logits=teacher_logits,
                labels=batch["labels"],
                student_hidden=student_out.get("hidden_states"),
            )

            loss = losses["total"] / self.config.gradient_accumulation_steps
            loss.backward()

            if (n_batches + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            for k in total_losses:
                if k in losses:
                    total_losses[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.student.eval()
        total_losses = {"total": 0}
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            teacher_out = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            student_out = self.student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            losses = self.criterion(
                student_out["logits"], teacher_out["logits"], batch["labels"],
            )
            total_losses["total"] += losses["total"].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    def _build_dataloader(self, data_path: str, shuffle: bool = True) -> DataLoader:
        """Build a DataLoader from JSONL data."""
        from src.training.lora_trainer import PersuasionDetectionDataset
        dataset = PersuasionDetectionDataset(
            data_path, self.tokenizer, self.config.max_seq_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )

    def _save_student(self, tag: str) -> None:
        """Save student model."""
        out = Path(self.config.output_dir) / tag
        out.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "config": {
                "model_name": self.config.student_model,
                "num_labels": self.config.num_labels,
            },
        }, str(out / "student_model.pt"))
        self.tokenizer.save_pretrained(str(out))
        logger.info(f"Student model saved: {out}")

    def _export_onnx(self) -> None:
        """Export student model to ONNX format."""
        out = Path(self.config.output_dir) / "onnx"
        out.mkdir(parents=True, exist_ok=True)

        self.student.eval()
        dummy = self.tokenizer(
            "This is a test sentence for export.",
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        onnx_path = out / "persuasix_student.onnx"

        # Wrapper for clean export
        class ExportWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                return self.model(input_ids, attention_mask)["logits"]

        wrapper = ExportWrapper(self.student)

        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=self.config.onnx_opset,
        )

        # Optimize ONNX
        if self.config.optimize_onnx:
            try:
                import onnx
                from onnxruntime.transformers import optimizer
                optimized_path = out / "persuasix_student_optimized.onnx"
                opt_model = optimizer.optimize_model(str(onnx_path), model_type="bert")
                opt_model.save_model_to_file(str(optimized_path))
                logger.info(f"Optimized ONNX: {optimized_path}")
            except ImportError:
                logger.info("onnxruntime not installed — skipping ONNX optimization")

        logger.success(f"ONNX export: {onnx_path}")

        # Save model card
        card = {
            "teacher": self.config.teacher_model,
            "student": self.config.student_model,
            "num_labels": self.config.num_labels,
            "max_seq_length": self.config.max_seq_length,
            "temperature": self.config.temperature,
            "format": "onnx",
            "opset": self.config.onnx_opset,
        }
        with open(out / "model_card.json", "w") as f:
            json.dump(card, f, indent=2)

    def _quantize_model(self) -> None:
        """Apply post-training quantization."""
        out = Path(self.config.output_dir) / "quantized"
        out.mkdir(parents=True, exist_ok=True)

        if self.config.quantize == "dynamic":
            quantized = torch.quantization.quantize_dynamic(
                self.student.cpu(),
                {nn.Linear},
                dtype=torch.qint8,
            )
            torch.save(quantized.state_dict(), str(out / "student_dynamic_int8.pt"))
            logger.info(f"Dynamic INT8 quantization saved: {out}")

        elif self.config.quantize == "int8":
            # Static quantization requires calibration
            self.student.cpu().eval()
            self.student.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            prepared = torch.quantization.prepare(self.student)
            # Calibration would happen here with representative data
            quantized = torch.quantization.convert(prepared)
            torch.save(quantized.state_dict(), str(out / "student_static_int8.pt"))
            logger.info(f"Static INT8 quantization saved: {out}")

        # Calculate compression stats
        original_size = sum(p.numel() * p.element_size() for p in self.student.parameters())
        logger.info(f"Original student size: {original_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model distillation for edge deployment")
    parser.add_argument("--teacher", type=str, default="roberta-large")
    parser.add_argument("--student", type=str, default="distilbert-base-uncased")
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--export-onnx", action="store_true", default=True)
    parser.add_argument("--quantize", choices=["none", "dynamic", "int8"], default="none")
    parser.add_argument("--output-dir", type=str, default="models/distilled")
    args = parser.parse_args()

    config = DistillationConfig(
        teacher_model=args.teacher,
        student_model=args.student,
        teacher_checkpoint=args.teacher_checkpoint,
        temperature=args.temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        export_onnx=args.export_onnx,
        quantize=args.quantize,
        output_dir=args.output_dir,
    )

    trainer = DistillationTrainer(config)
    history = trainer.train()
    logger.success("Distillation complete!")


if __name__ == "__main__":
    main()
