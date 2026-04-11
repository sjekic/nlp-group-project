"""
ad_timing.py
------------
Receptivity scoring and ad-slot recommendation policy.

Design
------
A single RECEPTIVITY SCORE ∈ [0, 1] is computed per 5-minute window.
Higher = more suitable for showing an ad.

Score components (weighted sum):

    1. Arousal penalty       — high arousal ↓ receptivity   (weight 0.40)
       Based on Breuer et al. (2021): low-to-moderate arousal maximises
       attention to sponsor messages.

    2. Valence penalty       — strong negative valence ↓ receptivity (weight 0.20)
       Ads shown during anger/grief create negative brand associations.

    3. Pressure penalty      — high on-pitch pressure ↓ receptivity  (weight 0.20)
       After intense attacking phases, a fatigue lull follows
       (Leifsson et al., 2024), but fans are still emotionally engaged
       DURING the pressure spike.

    4. Recent event penalty  — goal / red card in last window ↓ receptivity (weight 0.20)
       Fans need ~5 min to return to baseline after a high-intensity event
       (Pawlowski et al., 2024).

A window is flagged as AD_SAFE  if score ≥ 0.60,
                     AD_RISKY  if score ∈ [0.40, 0.60),
                     NO_AD     if score < 0.40.

Usage
-----
from src.ad_timing import score_windows, recommend_ad_slots

scored = score_windows(aligned_windows)
slots  = recommend_ad_slots(scored, max_ads_per_match=4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
W_AROUSAL  = 0.40
W_VALENCE  = 0.20
W_PRESSURE = 0.20
W_EVENT    = 0.20

# Thresholds
SAFE_THRESHOLD  = 0.60
RISKY_THRESHOLD = 0.40

# Pressure normalisation: values above this cap are treated as maximum pressure
PRESSURE_CAP = 50.0


def _normalise_pressure(pressure: pd.Series) -> pd.Series:
    """Clip and scale pressure to [0, 1]."""
    return (pressure.clip(upper=PRESSURE_CAP) / PRESSURE_CAP).fillna(0.0)


def score_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Compute receptivity scores for each window.

    Parameters
    ----------
    df : DataFrame from temporal_alignment.build_aligned_windows

    Returns
    -------
    Same DataFrame with additional columns:
        arousal_penalty, valence_penalty, pressure_penalty, event_penalty,
        receptivity_score, ad_label ("AD_SAFE" | "AD_RISKY" | "NO_AD")
    """
    result = df.copy()

    # ---- Component 1: Arousal penalty (high arousal → 1.0 penalty)
    # mean_arousal is already ∈ [0, 1]; default to 0.5 if classifier not run yet
    if "mean_arousal" in result.columns:
        result["arousal_penalty"] = result["mean_arousal"].clip(0, 1)
    else:
        result["arousal_penalty"] = 0.5  # neutral assumption

    # ---- Component 2: Valence penalty
    # Valence ∈ [-1, 1]; default to 0 (neutral) if not available
    if "mean_valence" in result.columns:
        result["valence_penalty"] = ((-result["mean_valence"]).clip(0, 1))
    else:
        result["valence_penalty"] = 0.0

    # ---- Component 3: Pressure penalty
    result["pressure_penalty"] = _normalise_pressure(result["mean_pressure"])

    # ---- Component 4: Recent high-intensity event penalty
    result["event_penalty"] = result["recent_high_intensity"].astype(float)

    # ---- Composite receptivity score (inverted penalty)
    raw_penalty = (
        W_AROUSAL  * result["arousal_penalty"]  +
        W_VALENCE  * result["valence_penalty"]  +
        W_PRESSURE * result["pressure_penalty"] +
        W_EVENT    * result["event_penalty"]
    )
    result["receptivity_score"] = (1.0 - raw_penalty).clip(0, 1)

    # ---- Ad label
    def _label(score):
        if score >= SAFE_THRESHOLD:
            return "AD_SAFE"
        elif score >= RISKY_THRESHOLD:
            return "AD_RISKY"
        else:
            return "NO_AD"

    result["ad_label"] = result["receptivity_score"].apply(_label)

    return result


def recommend_ad_slots(
    scored_df: pd.DataFrame,
    max_ads_per_match: int = 4,
    min_gap_windows: int = 2,
) -> pd.DataFrame:
    """Select the best non-overlapping ad windows per match.

    Parameters
    ----------
    scored_df : DataFrame from score_windows.
    max_ads_per_match : int
        Maximum ad slots to recommend per match.
    min_gap_windows : int
        Minimum number of 5-min windows between consecutive ads
        (default 2 → ads at least 10 min apart).

    Returns
    -------
    DataFrame of recommended ad slots with columns:
        fixture_id, match, window_5min, period_label,
        receptivity_score, ad_label, dominant_emotion,
        mean_arousal, mean_valence, mean_pressure
    """
    recommendations = []

    for fixture_id, group in scored_df.groupby("fixture_id"):
        # Only consider AD_SAFE windows
        candidates = (
            group[group["ad_label"] == "AD_SAFE"]
            .sort_values("receptivity_score", ascending=False)
            .copy()
        )

        selected = []
        for _, row in candidates.iterrows():
            # Enforce minimum gap
            if selected:
                gaps = [abs(row["window_5min"] - s["window_5min"]) for s in selected]
                if min(gaps) < min_gap_windows * 5:
                    continue
            selected.append(row)
            if len(selected) >= max_ads_per_match:
                break

        recommendations.extend(selected)

    cols = [
        "fixture_id", "match", "window_5min", "period_label",
        "receptivity_score", "ad_label", "dominant_emotion",
        "mean_arousal", "mean_valence", "mean_pressure",
        "tweet_count", "high_intensity_count", "recent_high_intensity"
    ]
    result = pd.DataFrame(recommendations)
    if result.empty:
        return result
    return result[[c for c in cols if c in result.columns]].sort_values(
        ["fixture_id", "window_5min"]
    ).reset_index(drop=True)


def summarise_policy(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a match-level summary of ad-timing policy outcomes."""
    summary = scored_df.groupby("fixture_id").agg(
        match=("match", "first"),
        total_windows=("window_5min", "count"),
        safe_windows=("ad_label", lambda x: (x == "AD_SAFE").sum()),
        risky_windows=("ad_label", lambda x: (x == "AD_RISKY").sum()),
        no_ad_windows=("ad_label", lambda x: (x == "NO_AD").sum()),
        mean_receptivity=("receptivity_score", "mean"),
        best_receptivity=("receptivity_score", "max"),
    ).reset_index()
    summary["pct_safe"] = (summary["safe_windows"] / summary["total_windows"] * 100).round(1)
    return summary
