"""CLI — Train PersuasiX models."""

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
from src.training.trainer import PersuasixTrainer
from src.utils.helpers import load_config, set_seed, get_device, count_parameters
from src.utils.visualization import plot_training_curves


@click.command()
@click.option("--config", default="config/config.yaml")
@click.option("--task", type=click.Choice(["detector", "explainer", "neutralizer", "all"]), default="all")
@click.option("--seed", default=42, type=int)
def main(config: str, task: str, seed: int):
    """Train one or all PersuasiX models."""
    cfg = load_config(config)
    set_seed(seed)
    device = get_device()
    logger.info(f"Device: {device}")

    data_dir = Path(cfg["data"]["processed_dir"])

    if task in ("detector", "all"):
        _train_detector(cfg, data_dir, device)

    if task in ("explainer", "all"):
        _train_seq2seq(cfg, data_dir, device, task_name="explainer")

    if task in ("neutralizer", "all"):
        _train_seq2seq(cfg, data_dir, device, task_name="neutralizer")


def _train_detector(cfg: dict, data_dir: Path, device: str):
    logger.info("=== Training Detector (RoBERTa) ===")

    model_cfg = cfg["detector"]
    model = PersuasionDetector(
        model_name=model_cfg["model_name"],
        num_labels=model_cfg["num_labels"],
        dropout=model_cfg["dropout"],
    )
    tokenizer = PersuasionDetector.get_tokenizer(model_cfg["model_name"])

    params = count_parameters(model)
    logger.info(f"Parameters: {params['trainable_millions']}M trainable / {params['total_millions']}M total")

    train_ds = PersuasixDataset(data_dir / "train.json", tokenizer, max_length=cfg["data"]["max_seq_length"])
    val_ds = PersuasixDataset(data_dir / "val.json", tokenizer, max_length=cfg["data"]["max_seq_length"])

    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"], num_workers=0)

    trainer = PersuasixTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=model_cfg["epochs"],
        learning_rate=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
        fp16=cfg["training"]["fp16"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        output_dir=str(Path(cfg["training"]["output_dir"]) / "detector"),
        task_type="classification",
        device=device,
    )

    history = trainer.train()
    plot_training_curves(history, save_path=str(Path(cfg["training"]["logging_dir"]) / "detector_curves.png"))
    model.save_pretrained(str(Path(cfg["training"]["output_dir"]) / "detector" / "final_model.pt"))
    logger.success("Detector training complete!")


def _train_seq2seq(cfg: dict, data_dir: Path, device: str, task_name: str):
    logger.info(f"=== Training {task_name.title()} (FLAN-T5) ===")

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_cfg = cfg[task_name]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["model_name"])

    DatasetClass = ExplainerDataset if task_name == "explainer" else NeutralizerDataset
    ds_kwargs = {"max_input_length": model_cfg["max_input_length"], "max_output_length": model_cfg["max_output_length"]}

    train_ds = DatasetClass(data_dir / "train.json", tokenizer, **ds_kwargs)
    val_ds = DatasetClass(data_dir / "val.json", tokenizer, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"], num_workers=0)

    trainer = PersuasixTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=model_cfg["epochs"],
        learning_rate=model_cfg["learning_rate"],
        fp16=cfg["training"]["fp16"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        output_dir=str(Path(cfg["training"]["output_dir"]) / task_name),
        task_type="seq2seq",
        device=device,
    )

    history = trainer.train()
    plot_training_curves(history, save_path=str(Path(cfg["training"]["logging_dir"]) / f"{task_name}_curves.png"))

    save_dir = Path(cfg["training"]["output_dir"]) / task_name / "final_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.success(f"{task_name.title()} training complete!")


if __name__ == "__main__":
    main()
