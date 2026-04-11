"""
evaluation.py
-------------
Metrics, visualisations, and summary statistics for the ad-timing pipeline.

Usage
-----
from src.evaluation import (
    plot_emotion_timeline,
    plot_receptivity_heatmap,
    plot_ad_slots_on_timeline,
    compute_emotion_event_correlation,
    print_summary,
)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ---- Consistent colour palette ----
EMOTION_COLORS = {
    "joy":      "#f9c74f",
    "anger":    "#f94144",
    "sadness":  "#4d9de0",
    "fear":     "#7b2d8b",
    "surprise": "#f3722c",
    "disgust":  "#90be6d",
}
AD_COLORS = {
    "AD_SAFE":  "#43aa8b",
    "AD_RISKY": "#f8961e",
    "NO_AD":    "#f94144",
}


# ---------------------------------------------------------------------------
# Timeline plots
# ---------------------------------------------------------------------------

def plot_emotion_timeline(
    window_df: pd.DataFrame,
    fixture_id,
    emotion_cols: list[str] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Line chart of emotion probabilities across match windows for one match."""
    match_data = window_df[window_df["fixture_id"] == fixture_id].sort_values("window_5min")
    if match_data.empty:
        raise ValueError(f"fixture_id {fixture_id} not found in window_df.")

    match_name = match_data["match"].iloc[0] if "match" in match_data.columns else str(fixture_id)

    if emotion_cols is None:
        emotion_cols = [c for c in ["joy", "anger", "sadness", "fear", "surprise", "disgust"]
                        if c in match_data.columns]

    fig, ax_ = (None, ax) if ax else plt.subplots(figsize=(14, 5))
    if fig is None:
        fig = ax_.get_figure()

    x = match_data["window_5min"]
    for emo in emotion_cols:
        ax_.plot(x, match_data[emo], label=emo.capitalize(),
                 color=EMOTION_COLORS.get(emo, None), linewidth=2, marker="o", markersize=3)

    # Half-time shading
    ax_.axvspan(45, 50, alpha=0.15, color="grey", label="Half-time")
    ax_.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    ax_.set_xlabel("Minute (relative to kick-off)", fontsize=11)
    ax_.set_ylabel("Mean emotion probability", fontsize=11)
    ax_.set_title(f"Emotion timeline — {match_name}", fontsize=13, fontweight="bold")
    ax_.legend(loc="upper right", fontsize=9)
    ax_.set_xlim(x.min(), x.max())
    ax_.set_ylim(0, 1)
    ax_.grid(axis="y", alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_receptivity_heatmap(
    scored_df: pd.DataFrame,
    top_n_matches: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of receptivity scores: matches × 5-min windows."""
    # Pick matches with most windows
    match_counts = scored_df.groupby("fixture_id")["window_5min"].count()
    top_fixtures = match_counts.nlargest(top_n_matches).index.tolist()
    subset = scored_df[scored_df["fixture_id"].isin(top_fixtures)].copy()

    pivot = subset.pivot_table(index="match", columns="window_5min",
                               values="receptivity_score", aggfunc="mean")
    pivot = pivot.fillna(np.nan)

    fig, ax = plt.subplots(figsize=(18, max(6, len(pivot) * 0.5)))
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Receptivity score"},
    )
    ax.set_title(f"Ad receptivity heatmap (top {top_n_matches} matches)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Match minute", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_ad_slots_on_timeline(
    scored_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
    fixture_id,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot receptivity score over time with recommended ad slots highlighted."""
    match_data   = scored_df[scored_df["fixture_id"] == fixture_id].sort_values("window_5min")
    match_recs   = recommended_df[recommended_df["fixture_id"] == fixture_id]
    match_name   = match_data["match"].iloc[0] if "match" in match_data.columns else str(fixture_id)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Background colour bands for ad label
    for _, row in match_data.iterrows():
        ax.axvspan(row["window_5min"], row["window_5min"] + 5,
                   alpha=0.15, color=AD_COLORS.get(row["ad_label"], "white"))

    ax.plot(match_data["window_5min"] + 2.5, match_data["receptivity_score"],
            color="#264653", linewidth=2.5, zorder=5)

    # Mark recommended slots
    for _, row in match_recs.iterrows():
        ax.axvline(row["window_5min"] + 2.5, color="#43aa8b", linewidth=2.5,
                   linestyle="--", zorder=10)
        ax.text(row["window_5min"] + 2.5, 1.02, "▼ AD", ha="center",
                fontsize=8, color="#43aa8b", fontweight="bold")

    ax.axhline(0.60, color="#43aa8b", linewidth=1, linestyle=":", alpha=0.7, label="Safe threshold")
    ax.axhline(0.40, color="#f8961e", linewidth=1, linestyle=":", alpha=0.7, label="Risky threshold")
    ax.axvspan(45, 50, alpha=0.15, color="grey")
    ax.axvline(0,  color="black", linewidth=1, linestyle="--", alpha=0.4)

    # Legend
    patches = [mpatches.Patch(color=v, alpha=0.4, label=k) for k, v in AD_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    ax.set_xlabel("Minute (relative to kick-off)", fontsize=11)
    ax.set_ylabel("Receptivity score", fontsize=11)
    ax.set_title(f"Ad-timing policy — {match_name}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(match_data["window_5min"].min(), match_data["window_5min"].max() + 5)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Correlation / metric utilities
# ---------------------------------------------------------------------------

def compute_emotion_event_correlation(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between emotion dimensions and match features."""
    numeric_cols = [
        "mean_arousal", "mean_valence", "mean_pressure",
        "high_intensity_count", "tweet_count", "receptivity_score"
    ]
    available = [c for c in numeric_cols if c in scored_df.columns]
    return scored_df[available].corr(method="pearson").round(3)


def compute_window_stats(scored_df: pd.DataFrame) -> dict:
    """Aggregate statistics over all windows."""
    stats = {
        "total_windows":  len(scored_df),
        "total_matches":  scored_df["fixture_id"].nunique(),
        "pct_safe":       (scored_df["ad_label"] == "AD_SAFE").mean() * 100,
        "pct_risky":      (scored_df["ad_label"] == "AD_RISKY").mean() * 100,
        "pct_no_ad":      (scored_df["ad_label"] == "NO_AD").mean() * 100,
        "mean_receptivity": scored_df["receptivity_score"].mean(),
        "mean_arousal":   scored_df["arousal"].mean(),
        "mean_valence":   scored_df["valence"].mean(),
    }
    return {k: round(v, 3) for k, v in stats.items()}


def print_summary(scored_df: pd.DataFrame, recommended_df: pd.DataFrame | None = None) -> None:
    stats = compute_window_stats(scored_df)
    print("=" * 60)
    print("AD-TIMING PIPELINE — SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<25} {v}")
    if recommended_df is not None:
        print(f"  {'recommended_slots':<25} {len(recommended_df)}")
        avg_score = recommended_df["receptivity_score"].mean()
        print(f"  {'avg_slot_receptivity':<25} {avg_score:.3f}")
    print("=" * 60)


def plot_label_distribution(scored_df: pd.DataFrame, save_path: str | None = None) -> plt.Figure:
    """Bar chart of AD_SAFE / AD_RISKY / NO_AD distribution."""
    counts = scored_df["ad_label"].value_counts().reindex(["AD_SAFE", "AD_RISKY", "NO_AD"])
    colors = [AD_COLORS[k] for k in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11)
    ax.set_ylabel("Number of windows")
    ax.set_title("Distribution of ad-timing labels across all windows",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
