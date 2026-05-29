"""CLI — Export trained models for deployment (ONNX, HuggingFace Hub)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import torch
from loguru import logger

from src.models.detector import PersuasionDetector
from src.utils.helpers import load_config


@click.command()
@click.option("--config", default="config/config.yaml")
@click.option("--detector-path", required=True, help="Path to detector checkpoint")
@click.option("--output-dir", default="models/exported")
@click.option("--format", "export_format", type=click.Choice(["onnx", "torchscript", "huggingface"]), default="onnx")
def main(config: str, detector_path: str, output_dir: str, export_format: str):
    """Export a trained detector model for deployment."""
    cfg = load_config(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {detector_path}")
    model = PersuasionDetector.from_pretrained(detector_path)
    model.eval()
    tokenizer = PersuasionDetector.get_tokenizer(cfg["detector"]["model_name"])

    if export_format == "onnx":
        _export_onnx(model, tokenizer, out, cfg)
    elif export_format == "torchscript":
        _export_torchscript(model, tokenizer, out, cfg)
    elif export_format == "huggingface":
        _export_huggingface(model, tokenizer, out)


def _export_onnx(model, tokenizer, out: Path, cfg: dict):
    logger.info("Exporting to ONNX …")
    dummy = tokenizer(
        "This is a test sentence.",
        max_length=cfg["data"]["max_seq_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    onnx_path = out / "persuasix_detector.onnx"
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    logger.success(f"ONNX model saved: {onnx_path}")


def _export_torchscript(model, tokenizer, out: Path, cfg: dict):
    logger.info("Exporting to TorchScript …")
    dummy = tokenizer(
        "This is a test sentence.",
        max_length=cfg["data"]["max_seq_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    traced = torch.jit.trace(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
    )
    ts_path = out / "persuasix_detector.pt"
    traced.save(str(ts_path))
    logger.success(f"TorchScript model saved: {ts_path}")


def _export_huggingface(model, tokenizer, out: Path):
    logger.info("Saving in HuggingFace format …")
    hf_path = out / "huggingface"
    hf_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(hf_path / "detector.pt"))
    tokenizer.save_pretrained(str(hf_path))
    logger.success(f"HuggingFace model saved: {hf_path}")


if __name__ == "__main__":
    main()
