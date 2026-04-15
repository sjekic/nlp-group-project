#!/usr/bin/env python3
"""Run pipeline stage 1: preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_tweets, load_match_events, build_pressure_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw tweet + match datasets.")
    parser.add_argument(
        "--tweet-path",
        type=Path,
        default=Path("data/premier_league_twitter_comments_match_windows_2020_07_09_to_2020_10_13.csv"),
        help="Path to raw tweet CSV.",
    )
    parser.add_argument(
        "--match-path",
        type=Path,
        default=Path("data/premier_league_combined_dataset_2020.csv"),
        help="Path to raw match CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for processed CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.tweet_path.exists():
        raise FileNotFoundError(f"Tweet file not found: {args.tweet_path}")
    if not args.match_path.exists():
        raise FileNotFoundError(f"Match file not found: {args.match_path}")

    tweets = load_tweets(str(args.tweet_path))
    match_events = load_match_events(str(args.match_path))
    pressure_windows = build_pressure_windows(match_events, window_size=5)

    tweets_out = args.output_dir / "tweets_cleaned.csv"
    match_out = args.output_dir / "match_events_cleaned.csv"
    pressure_out = args.output_dir / "pressure_windows.csv"

    tweets.to_csv(tweets_out, index=False)
    match_events.to_csv(match_out, index=False)
    pressure_windows.to_csv(pressure_out, index=False)

    print(f"[run_preprocessing] Saved: {tweets_out}")
    print(f"[run_preprocessing] Saved: {match_out}")
    print(f"[run_preprocessing] Saved: {pressure_out}")


if __name__ == "__main__":
    main()
