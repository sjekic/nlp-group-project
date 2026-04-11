"""
temporal_alignment.py
---------------------
Merges tweet emotion signals with match-event data into a single
time-windowed DataFrame per match.

Usage
-----
from src.temporal_alignment import build_aligned_windows

windows = build_aligned_windows(tweets_with_emotions, match_events, pressure_windows)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

WINDOW_SIZE = 5   # minutes


def aggregate_tweet_windows(tweet_df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Aggregate tweet-level emotion predictions into per-window averages.

    Parameters
    ----------
    tweet_df : DataFrame
        Output of EmotionClassifier.predict_df — must contain:
        fixture_id, window_5min, arousal, valence, dominant_emotion,
        and one column per emotion label.
    window_size : int
        Window size in minutes (should match the one used during preprocessing).

    Returns
    -------
    DataFrame with one row per (fixture_id, window_5min):
        tweet_count, mean_arousal, mean_valence, dominant_emotion_mode,
        + mean probability for each emotion label.
    """
    emotion_cols = [c for c in tweet_df.columns if c in
                    {"anger", "fear", "joy", "sadness", "surprise", "disgust"}]

    agg_dict = {"text_clean": "count"}   # tweet count

    # Only aggregate arousal/valence if the classifier has been run
    for col in ["arousal", "valence"] + emotion_cols:
        if col in tweet_df.columns:
            agg_dict[col] = "mean"

    agg = tweet_df.groupby(["fixture_id", "window_5min"]).agg(agg_dict).reset_index()
    agg.rename(columns={"text_clean": "tweet_count", "arousal": "mean_arousal", "valence": "mean_valence"}, inplace=True)

    # Dominant emotion per window (majority vote)
    dominant = (
        tweet_df.groupby(["fixture_id", "window_5min"])["dominant_emotion"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    agg = agg.merge(dominant, on=["fixture_id", "window_5min"], how="left")

    return agg


def attach_match_events(
    window_df: pd.DataFrame,
    match_df: pd.DataFrame,
    pressure_windows: pd.DataFrame,
    lookback_windows: int = 1,
) -> pd.DataFrame:
    """Attach match-level features to each tweet window.

    Parameters
    ----------
    window_df : DataFrame
        Output of aggregate_tweet_windows.
    match_df : DataFrame
        Output of preprocessing.load_match_events.
    pressure_windows : DataFrame
        Output of preprocessing.build_pressure_windows.
    lookback_windows : int
        How many preceding 5-min windows to include when flagging
        recent high-intensity events (default = 1 → last 5 min).

    Returns
    -------
    Merged DataFrame ready for the ad-timing model.
    """
    # 1. Attach pressure features
    merged = window_df.merge(pressure_windows, on=["fixture_id", "window_5min"], how="left")
    merged["mean_pressure"] = merged["mean_pressure"].fillna(0.0)
    merged["max_pressure"] = merged["max_pressure"].fillna(0.0)
    merged["high_intensity_count"] = merged["high_intensity_count"].fillna(0).astype(int)

    # 2. Flag whether a high-intensity event occurred in THIS or the
    #    preceding window(s) — fans need a few minutes to calm down.
    hi_events = match_df[match_df["is_high_intensity"]].copy()
    hi_events["window_5min"] = (hi_events["effective_minute"] // WINDOW_SIZE).astype(int) * WINDOW_SIZE

    def _recent_hi(row, hi_df, lookback):
        """Return 1 if any high-intensity event fell within lookback windows."""
        target_windows = [row["window_5min"] - i * WINDOW_SIZE for i in range(lookback + 1)]
        mask = (
            (hi_df["fixture_id"] == row["fixture_id"]) &
            (hi_df["window_5min"].isin(target_windows))
        )
        return int(hi_df[mask].shape[0] > 0)

    merged["recent_high_intensity"] = merged.apply(
        lambda r: _recent_hi(r, hi_events, lookback_windows), axis=1
    )

    # 3. Attach period label (1st half, 2nd half, extra time)
    period_map = (
        match_df[match_df["row_type"] == "period"]
        [["fixture_id", "period_label", "effective_minute"]]
        .dropna(subset=["period_label"])
    )
    # For each window assign the period that started most recently
    def _get_period(row, pm):
        fixture_periods = pm[pm["fixture_id"] == row["fixture_id"]]
        past = fixture_periods[fixture_periods["effective_minute"] <= row["window_5min"]]
        if past.empty:
            return "pre_match"
        return past.sort_values("effective_minute").iloc[-1]["period_label"]

    merged["period_label"] = merged.apply(lambda r: _get_period(r, period_map), axis=1)

    # 4. Attach match metadata (teams, kickoff)
    meta = match_df[["fixture_id", "match", "kickoff_utc", "home_team", "away_team", "derby",
                      "home_goals_final", "away_goals_final"]].drop_duplicates("fixture_id")
    merged = merged.merge(meta, on="fixture_id", how="left")

    return merged


def build_aligned_windows(
    tweet_df: pd.DataFrame,
    match_df: pd.DataFrame,
    pressure_windows: pd.DataFrame,
) -> pd.DataFrame:
    """End-to-end convenience wrapper.

    Parameters
    ----------
    tweet_df : DataFrame with emotion columns already attached.
    match_df : DataFrame from load_match_events.
    pressure_windows : DataFrame from build_pressure_windows.

    Returns
    -------
    One row per (fixture_id, window_5min) with all features needed
    for the ad-timing model.
    """
    window_df = aggregate_tweet_windows(tweet_df)
    aligned   = attach_match_events(window_df, match_df, pressure_windows)
    print(f"[build_aligned_windows] {len(aligned):,} windows across "
          f"{aligned['fixture_id'].nunique()} matches.")
    return aligned
