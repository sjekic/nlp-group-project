#!/usr/bin/env python3
"""Run pipeline stages 3-5: alignment, scoring, recommendations, evaluation tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temporal_alignment import build_aligned_windows
from src.ad_timing import score_windows, recommend_ad_slots, summarise_policy
from src.evaluation import compute_emotion_event_correlation, compute_window_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-inference pipeline stages.")
    parser.add_argument("--tweets-emotions", type=Path, default=Path("outputs/tweets_with_emotions.csv"))
    parser.add_argument("--match-events", type=Path, default=Path("outputs/match_events_cleaned.csv"))
    parser.add_argument("--pressure-windows", type=Path, default=Path("outputs/pressure_windows.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-ads-per-match", type=int, default=4)
    parser.add_argument("--min-gap-windows", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for p in [args.tweets_emotions, args.match_events, args.pressure_windows]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found: {p}")

    tweets = pd.read_csv(args.tweets_emotions)
    match_events = pd.read_csv(args.match_events)
    pressure_windows = pd.read_csv(args.pressure_windows)

    aligned = build_aligned_windows(tweets, match_events, pressure_windows)
    scored = score_windows(aligned)
    recommended = recommend_ad_slots(
        scored,
        max_ads_per_match=args.max_ads_per_match,
        min_gap_windows=args.min_gap_windows,
    )
    summary = summarise_policy(scored)
    corr = compute_emotion_event_correlation(scored)
    stats = compute_window_stats(scored)

    out_aligned = args.output_dir / "aligned_windows.csv"
    out_scored = args.output_dir / "scored_windows.csv"
    out_recommended = args.output_dir / "recommended_ad_slots.csv"
    out_summary = args.output_dir / "match_policy_summary.csv"
    out_corr = args.output_dir / "correlation_matrix.csv"
    out_stats = args.output_dir / "window_stats.csv"

    aligned.to_csv(out_aligned, index=False)
    scored.to_csv(out_scored, index=False)
    recommended.to_csv(out_recommended, index=False)
    summary.to_csv(out_summary, index=False)
    corr.to_csv(out_corr)
    pd.DataFrame([stats]).to_csv(out_stats, index=False)

    print(f"[run_postprocessing] Saved: {out_aligned}")
    print(f"[run_postprocessing] Saved: {out_scored}")
    print(f"[run_postprocessing] Saved: {out_recommended}")
    print(f"[run_postprocessing] Saved: {out_summary}")
    print(f"[run_postprocessing] Saved: {out_corr}")
    print(f"[run_postprocessing] Saved: {out_stats}")


if __name__ == "__main__":
    main()
