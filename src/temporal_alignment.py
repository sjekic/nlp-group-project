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

# All 28 GoEmotions labels -- used to detect emotion columns in the DataFrame
GO_EMOTIONS_LABELS = {
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral",
}


def aggregate_tweet_windows(tweet_df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Aggregate tweet-level emotion predictions into per-window averages.

    Parameters
    ----------
    tweet_df : DataFrame
        Output of EmotionClassifier.predict_df -- must contain:
        fixture_id, window_5min, arousal, valence, dominant_emotion,
        and one column per emotion label.
    window_size : int
        Window size in minutes (should match the one used during preprocessing).

    Returns
    -------
    DataFrame with one row per (fixture_id, window_5min):
        tweet_count, mean_arousal, mean_valence, dominant_emotion (mode),
        + mean probability for each emotion label present.
    """
    emotion_cols = [c for c in tweet_df.columns if c in GO_EMOTIONS_LABELS]

    agg_dict = {"text_clean": "count"}   # tweet count

    for col in ["arousal", "valence"] + emotion_cols:
        if col in tweet_df.columns:
            agg_dict[col] = "mean"

    agg = tweet_df.groupby(["fixture_id", "window_5min"]).agg(agg_dict).reset_index()
    agg.rename(
        columns={
            "text_clean": "tweet_count",
            "arousal": "mean_arousal",
            "valence": "mean_valence",
        },
        inplace=True,
    )

    # Dominant emotion per window (majority vote across tweets)
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
        recent high-intensity events (default = 1 -> last 5 min).

    Returns
    -------
    Merged DataFrame ready for the ad-timing model.
    """
    # 1. Attach pressure features
    merged = window_df.merge(pressure_windows, on=["fixture_id", "window_5min"], how="left")
    merged["mean_pressure"]        = merged["mean_pressure"].fillna(0.0)
    merged["max_pressure"]         = merged["max_pressure"].fillna(0.0)
    merged["high_intensity_count"] = merged["high_intensity_count"].fillna(0).astype(int)

    # 2. Flag recent high-intensity events -- vectorised
    hi_events = match_df[match_df["is_high_intensity"]].copy()
    hi_events["window_5min"] = (
        (hi_events["effective_minute"] // WINDOW_SIZE).astype(int) * WINDOW_SIZE
    )

    hi_lookup_rows = []
    for offset in range(lookback_windows + 1):
        shifted = hi_events[["fixture_id", "window_5min"]].copy()
        shifted["window_5min"] = shifted["window_5min"] + offset * WINDOW_SIZE
        hi_lookup_rows.append(shifted)

    hi_lookup = (
        pd.concat(hi_lookup_rows, ignore_index=True)
        .drop_duplicates()
        .assign(recent_high_intensity=1)
    )

    merged = merged.merge(hi_lookup, on=["fixture_id", "window_5min"], how="left")
    merged["recent_high_intensity"] = merged["recent_high_intensity"].fillna(0).astype(int)

    # 3. Attach period label -- vectorised with merge_asof
    period_map = (
        match_df[match_df["row_type"] == "period"]
        [["fixture_id", "period_label", "effective_minute"]]
        .dropna(subset=["period_label"])
        .rename(columns={"effective_minute": "window_5min"})
        .sort_values(["fixture_id", "window_5min"])
    )

    merged_sorted = merged.copy()
    # merge_asof requires identical dtypes and sorted keys on both sides.
    merged_sorted["window_5min"] = pd.to_numeric(
        merged_sorted["window_5min"], errors="coerce"
    ).astype(float)
    period_map["window_5min"] = pd.to_numeric(
        period_map["window_5min"], errors="coerce"
    ).astype(float)
    merged_sorted = merged_sorted.dropna(subset=["window_5min"])
    period_map = period_map.dropna(subset=["window_5min"])
    # For merge_asof, sort primarily by the "on" key, then by group key.
    merged_sorted = merged_sorted.sort_values(["window_5min", "fixture_id"]).reset_index(drop=True)
    period_map = period_map.sort_values(["window_5min", "fixture_id"]).reset_index(drop=True)

    if not period_map.empty:
        merged_sorted = pd.merge_asof(
            merged_sorted,
            period_map[["fixture_id", "window_5min", "period_label"]],
            on="window_5min",
            by="fixture_id",
            direction="backward",
        )
        merged_sorted["period_label"] = merged_sorted["period_label"].fillna("pre_match")
    else:
        merged_sorted["period_label"] = "pre_match"

    # 4. Attach match metadata
    meta_cols = [
        "fixture_id", "match", "kickoff_utc", "home_team", "away_team",
        "derby", "home_goals_final", "away_goals_final",
    ]
    available_meta = [c for c in meta_cols if c in match_df.columns]
    meta = match_df[available_meta].drop_duplicates("fixture_id")
    result = merged_sorted.merge(meta, on="fixture_id", how="left")

    return result


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
    print(
        f"[build_aligned_windows] {len(aligned):,} windows across "
        f"{aligned['fixture_id'].nunique()} matches."
    )
    return aligned
