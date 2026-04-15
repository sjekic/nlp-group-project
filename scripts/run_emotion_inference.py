#!/usr/bin/env python3
"""Run pipeline stage 2: GPU emotion inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.emotion_classifier import EmotionClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run emotion inference on cleaned tweets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/tweets_cleaned.csv"),
        help="Input CSV with cleaned tweets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tweets_with_emotions.csv"),
        help="Output CSV with emotion columns.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size. Lower if CUDA OOM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Force execution device. Default: auto-detect.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text_clean",
        help="Input text column name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in input. Available: {list(df.columns)}")

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device or auto_device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this node.")

    print(f"[run_emotion_inference] Loaded {len(df):,} rows from {args.input}")
    print(f"[run_emotion_inference] Using device={device}, batch_size={args.batch_size}")

    clf = EmotionClassifier(device=device)
    out_df = clf.predict_df(df, text_col=args.text_col, batch_size=args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"[run_emotion_inference] Saved {len(out_df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
