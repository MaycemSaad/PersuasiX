"""Training callbacks: early stopping and model checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def on_epoch_end(self, epoch: int, val_loss: float, model: nn.Module) -> str | None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return "stop"
        return None


class ModelCheckpoint:
    """Save the model when a monitored metric improves."""

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value: float | None = None

    def on_epoch_end(self, epoch: int, val_loss: float, model: nn.Module) -> str | None:
        current = val_loss
        is_better = (
            self.best_value is None
            or (self.mode == "min" and current < self.best_value)
            or (self.mode == "max" and current > self.best_value)
        )

        if is_better:
            self.best_value = current
            path = self.save_dir / f"best_model_epoch{epoch}.pt"
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(path))
            else:
                torch.save(model.state_dict(), path)
            logger.info(f"Checkpoint saved: {path} ({self.monitor}={current:.4f})")

        return None
