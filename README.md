# Predicting Emotions Among Football Fans to Optimize Advertising Using NLP

**IE University — NLP Course Project | Group 6**

> Simonida Jekic · Vako Khvedelidze · Michail Sifakis · Sofiia Avetisian

---

## Overview

This project builds a context-aware ad-timing system for live Premier League broadcasts. Rather than relying on fixed ad schedules, we use NLP to extract fan emotions from Twitter in real time and combine them with structured match-event data (goals, cards, pressure, whistles) to identify windows where viewers are most receptive to advertising.

The key insight comes from Breuer et al. (2021): **low-to-moderate arousal + valence-neutral states maximise viewer attention to sponsor messages**, while high arousal (either positive or negative) pushes attention away from ads.

---

## Pipeline

```
Raw Data
  ├── Twitter CSV  (14 492 tweets · 42 matches · 2020 season)
  └── Match CSV    (79 569 rows  · 380 matches · 2024/25 season)
        │
        ▼
1. Preprocessing            notebooks/01_preprocessing.ipynb
        │
        ▼
2. Emotion Classification   notebooks/02_emotion_classification.ipynb
   (RoBERTa / cardiffnlp)
        │
        ▼
3. Temporal Alignment       notebooks/03_temporal_alignment.ipynb
   Tweet windows ↔ Match events
        │
        ▼
4. Ad-Timing Model          notebooks/04_ad_timing_model.ipynb
   Rule-based + scoring policy
        │
        ▼
5. Evaluation & Visuals     notebooks/05_evaluation.ipynb
```

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── premier_league_combined_dataset_2024_2025.csv
│   └── premier_league_twitter_comments_match_windows_2020_07_09_to_2020_10_13.csv
├── src/
│   ├── preprocessing.py          Tweet & match preprocessing utilities
│   ├── emotion_classifier.py     HuggingFace emotion inference wrapper
│   ├── temporal_alignment.py     Window builder & event-tweet merger
│   ├── ad_timing.py              Receptivity scorer & ad-slot policy
│   └── evaluation.py             Metrics, plots, summary stats
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_emotion_classification.ipynb
│   ├── 03_temporal_alignment.ipynb
│   ├── 04_ad_timing_model.ipynb
│   └── 05_evaluation.ipynb
└── outputs/                      Generated figures & result CSVs
```

---

## Setup

```bash
pip install -r requirements.txt
```

All notebooks are designed to run on **Google Colab** (free tier). Mount your Drive and set `DATA_DIR` at the top of each notebook.

## Cluster Execution (Slurm)

This repo includes script-based pipeline runners and Slurm templates:

- `scripts/run_preprocessing.py` (CPU)
- `scripts/run_emotion_inference.py` (GPU)
- `scripts/run_postprocessing.py` (CPU)
- `scripts/slurm_01_preprocess_cpu.sbatch`
- `scripts/slurm_02_emotion_gpu.sbatch`
- `scripts/slurm_03_postprocess_cpu.sbatch`
- `scripts/submit_cluster_pipeline.sh`

Detailed step-by-step instructions are in:

- `docs/UNIVERSITY_CLUSTER_RUNBOOK.md`

### What We Actually Did on the IE Cluster (Simple Summary)

We ran the pipeline on the university HPC cluster (`rust` login node + Slurm jobs on `haskell`).

1. Created and activated a Conda environment:
   - `conda create -n nlp-project python=3.11 -y`
   - `conda activate nlp-project`
2. Copied this repo from laptop to cluster with `rsync`.
3. Submitted the 3-stage pipeline with Slurm:
   - Stage 1 (CPU): preprocessing
   - Stage 2 (GPU): emotion inference
   - Stage 3 (CPU): alignment + scoring + summaries
4. Checked status with:
   - `squeue -u $USER`
   - `sacct -u <user> --format=JobID,JobName,State,ExitCode,Elapsed --starttime today`
5. Outcome:
   - `nlp-preprocess` completed
   - `nlp-emotion` completed
   - first `nlp-postprocess` run failed due to a pandas merge key issue (`merge_asof` key dtype/sort)
6. Fix applied:
   - updated `src/temporal_alignment.py` to enforce compatible `window_5min` types and sorting before `merge_asof`
   - re-submitted postprocess job
7. Final result:
   - postprocess completed successfully and all final output CSVs were generated.

Useful final outputs:
- `outputs/tweets_with_emotions.csv`
- `outputs/aligned_windows.csv`
- `outputs/scored_windows.csv`
- `outputs/recommended_ad_slots.csv`
- `outputs/match_policy_summary.csv`
- `outputs/correlation_matrix.csv`
- `outputs/window_stats.csv`

---

## Data Sources

| Dataset | Source |
|---|---|
| Twitter (EPL 2020) | [Kaggle — wjia26/epl-teams-twitter-sentiment-dataset](https://www.kaggle.com/datasets/wjia26/epl-teams-twitter-sentiment-dataset) |
| Match events (2024/25) | [SportMonks Football API](https://my.sportmonks.com) |

---

## Key Design Decisions

- **No heavy text preprocessing** — RoBERTa tokenizer handles normalisation internally; removing stopwords would strip contextual cues.
- **Emotion, not sentiment** — We use `cardiffnlp/twitter-roberta-base-emotion` for multi-class emotion labels (joy, anger, sadness, fear, surprise, disgust) rather than a binary positive/negative score.
- **5-minute windows** — Tweets are bucketed into 5-minute intervals per match to smooth noise while preserving temporal dynamics.
- **Receptivity score** — Combines (a) low arousal-emotion dominance, (b) low match pressure, and (c) absence of recent high-intensity events (goals, red cards) into a single [0–1] score per window.

---

## References

1. Y. Xu, *Scalable Computing*, 2023. doi:10.12694/scpe.v24i3.2342  
2. F. Wunderlich & D. Memmert, *SNAM*, 2022. doi:10.1007/s13278-021-00842-z  
3. M. Ortu & F. Mola, *Computational Statistics*, 2025. doi:10.1007/s00180-024-01584-0  
4. J. A. Hernández-Aguilar & Y. Calderón-Segura, *Discover Computing*, 2024. doi:10.1007/s42979-024-03401-3  
5. C. Breuer, C. Rumpf & F. Boronczyk, *Psychology & Marketing*, 2021. doi:10.1002/mar.21481  
6. M. Mohr, P. Krustrup & J. Bangsbo, *J. Sports Sciences*, 2003. doi:10.1080/0264041031000071182  
7. E. N. Leifsson et al., *J. Sports Sciences*, 2024. doi:10.1080/02640414.2024.2364135  
8. T. Pawlowski et al., *J. Economic Behavior & Organization*, 2024. doi:10.1016/j.jebo.2024.04.018
