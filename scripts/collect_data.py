"""CLI — Stage 1: Collect raw data from all sources."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from loguru import logger

from src.data.collector import DataCollector
from src.data.cleaner import DataCleaner
from src.utils.helpers import load_config


@click.command()
@click.option("--config", default="config/config.yaml", help="Path to config file")
@click.option("--output-dir", default=None, help="Override output directory")
def main(config: str, output_dir: str | None):
    """Collect and clean raw data for PersuasiX."""
    cfg = load_config(config)
    out_dir = output_dir or cfg["data"]["raw_dir"]

    logger.info("=== PersuasiX Data Collection ===")

    collector = DataCollector(
        output_dir=out_dir,
        rate_limit=cfg["collection"]["rate_limit_seconds"],
    )
    raw_df = collector.collect_all()

    if raw_df.empty:
        logger.error("No data collected. Exiting.")
        return

    cleaner = DataCleaner(
        min_length=cfg["collection"]["min_text_length"],
        max_length=cfg["collection"]["max_text_length"],
    )
    clean_df = cleaner.clean(raw_df)

    out_path = Path(out_dir) / "clean_collected.json"
    clean_df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    logger.success(f"Clean data saved: {out_path} ({len(clean_df)} rows)")


if __name__ == "__main__":
    main()
