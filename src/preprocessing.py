"""
preprocessing.py
----------------
Utilities to clean and prepare both the Twitter and match-event datasets
for downstream NLP and temporal analysis.

Usage
-----
from src.preprocessing import load_tweets, load_match_events

tweets = load_tweets("data/premier_league_twitter_comments_match_windows_2020_07_09_to_2020_10_13.csv")
match  = load_match_events("data/premier_league_combined_dataset_2024_2025.csv")
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Emotions with HIGH arousal — bad windows for ads (Breuer et al., 2021)
HIGH_AROUSAL_EMOTIONS = {"anger", "fear", "surprise"}

# Minimum tweet length (chars) after cleaning
MIN_TWEET_LENGTH = 10

# Match window: 30 min before kick-off up to 120 min (incl. extra time)
WINDOW_START_MIN = -30
WINDOW_END_MIN   = 120


# ---------------------------------------------------------------------------
# Tweet helpers
# ---------------------------------------------------------------------------

def _clean_tweet_text(text: str) -> str:
    """Remove URLs, mentions, hashtag symbols, and extra whitespace.
    We keep the actual words from hashtags (e.g. #Arsenal → Arsenal)
    because they carry semantic content that RoBERTa can use.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # hashtag → word
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_tweets(path: str, window_minutes: tuple = (WINDOW_START_MIN, WINDOW_END_MIN)) -> pd.DataFrame:
    """Load and minimally preprocess the Twitter dataset.

    Parameters
    ----------
    path : str
        Path to the raw CSV.
    window_minutes : tuple
        (start, end) relative-minute range to keep.

    Returns
    -------
    pd.DataFrame with columns:
        fixture_id, match, kickoff_utc, created_at,
        relative_minute_from_kickoff, text, text_clean,
        polarity, home_team, away_team, derby
    """
    df = pd.read_csv(path, parse_dates=["kickoff_utc", "created_at", "window_start_utc", "window_end_utc"])

    # Drop rows without text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() >= MIN_TWEET_LENGTH].copy()

    # Filter to match-window
    start, end = window_minutes
    df = df[
        (df["relative_minute_from_kickoff"] >= start) &
        (df["relative_minute_from_kickoff"] <= end)
    ].copy()

    # Minimal text cleaning
    df["text_clean"] = df["text"].apply(_clean_tweet_text)

    # Drop rows where cleaning wiped all content
    df = df[df["text_clean"].str.len() >= MIN_TWEET_LENGTH].copy()

    # Derive 5-minute bucket for each tweet (relative to kick-off)
    df["window_5min"] = (df["relative_minute_from_kickoff"] // 5).astype(int) * 5

    # Keep only the columns used downstream
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

# Events that cause high audience arousal — penalised in the receptivity score
HIGH_INTENSITY_EVENTS = {"Goal", "Redcard", "Penalty", "Own Goal", "Missed Penalty", "VAR_CARD"}

# Events that are neutral/administrative
NEUTRAL_EVENTS = {"Substitution", "Yellowcard", "Yellow/Red card"}


def load_match_events(path: str) -> pd.DataFrame:
    """Load and structure a combined match dataset.

    Compatible with both:
      - premier_league_combined_dataset_2020.csv  (2020 season, no pressure)
      - premier_league_combined_dataset_2024_2025.csv  (2024/25 season, has pressure)

    Returns a DataFrame with one row per meaningful match event,
    including period boundaries, pressure values, and event types.
    """
    parse_cols = ["kickoff_utc"]
    # whistle_utc only exists in 2020 combined dataset
    raw_df = pd.read_csv(path, nrows=0)
    if "whistle_utc" in raw_df.columns:
        parse_cols.append("whistle_utc")
    df = pd.read_csv(path, parse_dates=parse_cols)

    # Normalise minute to float
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df["extra_minute"] = pd.to_numeric(df["extra_minute"], errors="coerce").fillna(0)

    # Effective match minute (handles extra time)
    df["effective_minute"] = df["minute"].fillna(0) + df["extra_minute"]

    # Flag high-intensity events
    df["is_high_intensity"] = df["event_type_name"].isin(HIGH_INTENSITY_EVENTS)

    # Pressure is already numeric — fill NaN with 0 for non-pressure rows
    df["pressure_value"] = pd.to_numeric(df["pressure_value"], errors="coerce").fillna(0.0)

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
        "whistle_type", "whistle_utc", "time_added_minutes",
        "row_sequence_in_match"
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    print(f"[load_match_events] {len(df):,} rows across {df['fixture_id'].nunique()} matches.")
    return df


def build_pressure_windows(match_df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Aggregate match pressure into fixed-minute windows per match.

    Parameters
    ----------
    match_df : DataFrame from load_match_events
    window_size : int, minutes per window

    Returns
    -------
    DataFrame with columns: fixture_id, window_5min, mean_pressure,
        max_pressure, high_intensity_count
    """
    pressure_rows = match_df[match_df["row_type"] == "pressure"].copy()
    pressure_rows["window_5min"] = (pressure_rows["effective_minute"] // window_size).astype(int) * window_size

    agg = pressure_rows.groupby(["fixture_id", "window_5min"]).agg(
        mean_pressure=("pressure_value", "mean"),
        max_pressure=("pressure_value", "max"),
    ).reset_index()

    # High-intensity event counts per window
    hi_events = match_df[match_df["is_high_intensity"]].copy()
    hi_events["window_5min"] = (hi_events["effective_minute"] // window_size).astype(int) * window_size
    hi_count = hi_events.groupby(["fixture_id", "window_5min"]).size().reset_index(name="high_intensity_count")

    result = agg.merge(hi_count, on=["fixture_id", "window_5min"], how="left")
    result["high_intensity_count"] = result["high_intensity_count"].fillna(0).astype(int)

    return result
