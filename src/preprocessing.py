"""
preprocessing.py
----------------
Utilities to clean and prepare both the Twitter and match-event datasets
for downstream NLP and temporal analysis.

Usage
-----
from src.preprocessing import load_tweets, load_match_events, build_pressure_windows

tweets = load_tweets("data/premier_league_twitter_comments_match_windows_2020_07_09_to_2020_10_13.csv")
match  = load_match_events("data/premier_league_combined_dataset_2020.csv")
pressure_windows = build_pressure_windows(match)
"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_TWEET_LENGTH = 10
WINDOW_START_MIN = -30
WINDOW_END_MIN   = 120

# Events that cause high audience arousal
HIGH_INTENSITY_EVENTS = {"Goal", "Redcard", "Penalty", "Own Goal", "Missed Penalty", "VAR_CARD"}

# Event intensity weights — used to build a pressure proxy for the 2020 dataset
# (which has no SportMonks pressure values)
EVENT_INTENSITY_WEIGHTS = {
    "Goal":           3.0,
    "Penalty":        2.5,
    "Missed Penalty": 2.0,
    "Own Goal":       3.0,
    "Redcard":        2.5,
    "VAR_CARD":       2.0,
    "Yellowcard":     1.0,
    "Yellow/Red card":1.5,
    "Substitution":   0.3,
}

NEUTRAL_EVENTS = {"Substitution", "Yellowcard", "Yellow/Red card"}


# ---------------------------------------------------------------------------
# Tweet helpers
# ---------------------------------------------------------------------------

def _clean_tweet_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_tweets(path: str, window_minutes: tuple = (WINDOW_START_MIN, WINDOW_END_MIN)) -> pd.DataFrame:
    """Load and minimally preprocess the Twitter dataset."""
    df = pd.read_csv(path, parse_dates=["kickoff_utc", "created_at",
                                         "window_start_utc", "window_end_utc"])
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() >= MIN_TWEET_LENGTH].copy()

    start, end = window_minutes
    df = df[
        (df["relative_minute_from_kickoff"] >= start) &
        (df["relative_minute_from_kickoff"] <= end)
    ].copy()

    df["text_clean"] = df["text"].apply(_clean_tweet_text)
    df = df[df["text_clean"].str.len() >= MIN_TWEET_LENGTH].copy()
    df["window_5min"] = (df["relative_minute_from_kickoff"] // 5).astype(int) * 5

    keep = [
        "fixture_id", "match", "kickoff_utc", "created_at",
        "relative_minute_from_kickoff", "window_5min",
        "text", "text_clean", "polarity",
        "home_team", "away_team", "derby"
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    print(f"[load_tweets] {len(df):,} tweets across {df['fixture_id'].nunique()} matches after filtering.")
    return df


# ---------------------------------------------------------------------------
# Match-event helpers
# ---------------------------------------------------------------------------

def load_match_events(path: str) -> pd.DataFrame:
    """Load and structure a combined match dataset.

    Compatible with both:
      - premier_league_combined_dataset_2020.csv  (2020 season, no pressure)
      - premier_league_combined_dataset_2024_2025.csv  (2024/25 season, has pressure)
    """
    parse_cols = ["kickoff_utc"]
    raw_df = pd.read_csv(path, nrows=0)
    if "whistle_utc" in raw_df.columns:
        parse_cols.append("whistle_utc")
    df = pd.read_csv(path, parse_dates=parse_cols)

    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df["extra_minute"] = pd.to_numeric(df["extra_minute"], errors="coerce").fillna(0)
    df["effective_minute"] = df["minute"].fillna(0) + df["extra_minute"]
    df["is_high_intensity"] = df["event_type_name"].isin(HIGH_INTENSITY_EVENTS)

    # pressure_value: use native column if present (2024/25), else 0
    if "pressure_value" in df.columns:
        df["pressure_value"] = pd.to_numeric(df["pressure_value"], errors="coerce").fillna(0.0)
    else:
        df["pressure_value"] = 0.0

    keep = [
        "fixture_id", "match", "kickoff_utc", "gameweek",
        "home_team", "away_team", "derby",
        "row_type", "effective_minute", "minute", "extra_minute", "minute_label",
        "event_category", "event_type_name", "event_type_code",
        "participant_name", "participant_location",
        "player_name", "related_player_name",
        "info", "period_label",
        "pressure_value", "is_high_intensity",
        "home_goals_final", "away_goals_final",
        "whistle_type", "time_added_minutes",
        "first_half_added_minutes", "second_half_added_minutes",
        "row_sequence_in_match"
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    print(f"[load_match_events] {len(df):,} rows across {df['fixture_id'].nunique()} matches.")
    return df


def build_pressure_windows(match_df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Build per-window match intensity features.

    For datasets WITH native pressure values (2024/25): aggregates pressure_value.
    For datasets WITHOUT pressure (2020): derives an intensity proxy from
    event density, weighted by event type (goals=3.0, red cards=2.5, etc.).

    Returns DataFrame with columns:
        fixture_id, window_5min, mean_pressure, max_pressure, high_intensity_count
    """
    has_native_pressure = (match_df["pressure_value"] > 0).any()

    event_rows = match_df[match_df["row_type"] == "event"].copy()
    event_rows["window_5min"] = (event_rows["effective_minute"] // window_size).astype(int) * window_size

    if has_native_pressure:
        # 2024/25 path — aggregate real pressure values
        pressure_rows = match_df[match_df["row_type"] == "pressure"].copy()
        pressure_rows["window_5min"] = (pressure_rows["effective_minute"] // window_size).astype(int) * window_size
        agg = pressure_rows.groupby(["fixture_id", "window_5min"]).agg(
            mean_pressure=("pressure_value", "mean"),
            max_pressure=("pressure_value", "max"),
        ).reset_index()
    else:
        # 2020 path — derive intensity proxy from weighted event counts
        event_rows["intensity_weight"] = event_rows["event_type_name"].map(
            EVENT_INTENSITY_WEIGHTS
        ).fillna(0.5)

        agg = event_rows.groupby(["fixture_id", "window_5min"]).agg(
            mean_pressure=("intensity_weight", "sum"),   # total weight = proxy for intensity
            max_pressure=("intensity_weight", "max"),
        ).reset_index()

        # Build a full window grid (every 5-min window for every match)
        # so windows with no events get 0 pressure (not NaN)
        all_fixtures = match_df["fixture_id"].unique()
        all_windows  = range(-30, 121, window_size)
        full_grid = pd.DataFrame(
            [(f, w) for f in all_fixtures for w in all_windows],
            columns=["fixture_id", "window_5min"]
        )
        agg = full_grid.merge(agg, on=["fixture_id", "window_5min"], how="left").fillna(0)

    # High-intensity event counts per window (both paths)
    hi_events = match_df[match_df["is_high_intensity"]].copy()
    hi_events["window_5min"] = (hi_events["effective_minute"] // window_size).astype(int) * window_size
    hi_count = hi_events.groupby(["fixture_id", "window_5min"]).size().reset_index(name="high_intensity_count")

    result = agg.merge(hi_count, on=["fixture_id", "window_5min"], how="left")
    result["high_intensity_count"] = result["high_intensity_count"].fillna(0).astype(int)

    print(f"[build_pressure_windows] {len(result):,} windows | native_pressure={has_native_pressure}")
    return result
