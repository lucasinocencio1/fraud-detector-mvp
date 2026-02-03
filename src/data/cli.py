import argparse
from pathlib import Path

import pandas as pd

from src.data.logging import configure_logging
from src.data.pipelines.features import run_feature_pipeline
from src.data.pipelines.ingest import append_batch
from src.data.pipelines.split import run_split

logger = configure_logging(name="fraud_data.cli")


def cmd_ingest(args: argparse.Namespace) -> None:
    batch_path = Path(args.batch_path)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")
    batch_df = pd.read_csv(batch_path)
    append_batch(batch_df)


def cmd_split(_: argparse.Namespace) -> None:
    run_split()


def cmd_features(_: argparse.Namespace) -> None:
    run_feature_pipeline()


def cmd_full_run(args: argparse.Namespace) -> None:
    if args.batch_path:
        cmd_ingest(args)
    cmd_split(args)
    cmd_features(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fraud detector data pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Append a batch of raw transactions")
    ingest_parser.add_argument("batch_path", help="Path to CSV batch file")
    ingest_parser.set_defaults(func=cmd_ingest)

    split_parser = subparsers.add_parser("split", help="Split raw transactions into train/val/test")
    split_parser.set_defaults(func=cmd_split)

    features_parser = subparsers.add_parser("features", help="Build feature artifacts")
    features_parser.set_defaults(func=cmd_features)

    full_parser = subparsers.add_parser("full-run", help="Run split and feature pipeline, optionally ingesting a batch first")
    full_parser.add_argument("--batch-path", help="Optional raw batch CSV to ingest before running pipelines")
    full_parser.set_defaults(func=cmd_full_run)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
