"""Unified training loop for all PersuasiX tasks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint
from ..utils.helpers import get_device


class PersuasixTrainer:
    """
    Training loop supporting:
      - Multi-label classification (detector)
      - Seq2seq generation (explainer, neutralizer)
      - Mixed-precision training
      - Gradient accumulation
      - Callbacks (early stopping, checkpointing)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        fp16: bool = True,
        output_dir: str = "checkpoints",
        task_type: str = "classification",
        device: str | None = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16 and torch.cuda.is_available()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = task_type

        self.optimizer = optimizer or AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * epochs
        self.scheduler = scheduler or CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )

        self.scaler = torch.amp.GradScaler("cuda") if self.fp16 else None

        self.callbacks = [
            EarlyStopping(patience=3, min_delta=0.001),
            ModelCheckpoint(
                save_dir=str(self.output_dir),
                monitor="val_loss",
                mode="min",
            ),
        ]

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_time": [],
        }

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop."""
        logger.info(
            f"Starting training: {self.epochs} epochs | "
            f"device={self.device} | fp16={self.fp16} | "
            f"task={self.task_type}"
        )

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            elapsed = time.time() - start

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(lr)
            self.history["epoch_time"].append(elapsed)

            logger.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                f"lr: {lr:.2e} | time: {elapsed:.1f}s"
            )

            should_stop = False
            for cb in self.callbacks:
                cb_result = cb.on_epoch_end(epoch, val_loss, self.model)
                if cb_result == "stop":
                    should_stop = True

            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        return self.history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"Train epoch {epoch}", leave=False)
        self.optimizer.zero_grad()

        for step, batch in enumerate(progress):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.fp16:
                with torch.amp.autocast("cuda"):
                    outputs = self._forward(batch)
                    loss = outputs["loss"] / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self._forward(batch)
                loss = outputs["loss"] / self.gradient_accumulation_steps
                loss.backward()

            total_loss += outputs["loss"].item()
            num_batches += 1

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            progress.set_postfix(loss=f"{outputs['loss'].item():.4f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc=f"Val epoch {epoch}", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self._forward(batch)
            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.task_type == "classification":
            return self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        else:
            return self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
