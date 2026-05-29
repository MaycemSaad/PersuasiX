"""CLI — Build the full PersuasiX dataset: collect → clean → enrich → validate → split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
from loguru import logger

from src.data.collector import DataCollector
from src.data.cleaner import DataCleaner
from src.data.enricher import LLMEnricher
from src.data.validator import DataValidator
from src.utils.helpers import load_config, set_seed


@click.command()
@click.option("--config", default="config/config.yaml", help="Path to config file")
@click.option("--skip-enrichment", is_flag=True, help="Skip LLM enrichment (use fallback)")
@click.option("--seed", default=42, type=int)
def main(config: str, skip_enrichment: bool, seed: int):
    """Build the complete PersuasiX dataset through the 4-stage pipeline."""
    cfg = load_config(config)
    set_seed(seed)

    logger.info("=" * 50)
    logger.info("  PersuasiX Dataset Builder")
    logger.info("=" * 50)

    # Stage 1: Collect
    logger.info("Stage 1/4: Collecting data …")
    collector = DataCollector(output_dir=cfg["data"]["raw_dir"])
    raw_df = collector.collect_all()

    # Stage 2: Clean
    logger.info("Stage 2/4: Cleaning data …")
    cleaner = DataCleaner(
        min_length=cfg["collection"]["min_text_length"],
        max_length=cfg["collection"]["max_text_length"],
    )
    clean_df = cleaner.clean(raw_df)

    # Stage 3: Enrich
    logger.info("Stage 3/4: Enriching data …")
    if skip_enrichment:
        logger.info("Skipping LLM enrichment — using fallback")
        enricher = LLMEnricher(provider="none")
        enricher._client = None
    else:
        enricher = LLMEnricher(
            provider=cfg["enrichment"]["provider"],
            model=cfg["enrichment"]["model"],
            batch_size=cfg["enrichment"]["batch_size"],
            temperature=cfg["enrichment"]["temperature"],
        )
    enriched_df = enricher.enrich(clean_df)

    # Stage 4: Validate
    logger.info("Stage 4/4: Validating data …")
    validator = DataValidator(use_embeddings=not skip_enrichment)
    validated_df = validator.validate(enriched_df)

    # Split
    logger.info("Splitting dataset …")
    train_ratio = cfg["data"]["train_split"]
    val_ratio = cfg["data"]["val_split"]

    validated_df = validated_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(validated_df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": validated_df.iloc[:train_end],
        "val": validated_df.iloc[train_end:val_end],
        "test": validated_df.iloc[val_end:],
    }

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in splits.items():
        path = out_dir / f"{name}.json"
        split_df.to_json(path, orient="records", lines=True, force_ascii=False)
        logger.info(f"  {name}: {len(split_df)} rows → {path}")

    full_path = out_dir / "full_dataset.json"
    validated_df.to_json(full_path, orient="records", lines=True, force_ascii=False)

    logger.success(f"Dataset build complete! Total: {n} rows")
    logger.info(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")


if __name__ == "__main__":
    main()
