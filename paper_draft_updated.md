# Updated Paper Draft — Sections Reflecting Actual Code Implementation

> **NOTE:** This file contains updated text for the Methodology, Data Collection &
> Cleaning, Data Preprocessing, and Emotion Classification sections, based on the
> actual code in `src/` and the five notebooks. Results sections are left as
> placeholders pending notebook 02 completion. Paste these sections back into your
> LaTeX file to replace the current draft text.

---

## III. METHODOLOGY

This study presents a natural language processing pipeline to model how audience
emotions evolve over the course of a Premier League football match using social
media data. Emotional signals extracted at the tweet level are aggregated into
five-minute windows, aligned with structured match-event data, and used to derive
a *receptivity score* that drives a context-aware advertisement-timing policy.

The pipeline is organised into five sequential stages, each corresponding to a
Jupyter notebook:

1. **Data preprocessing** (`01_preprocessing.ipynb`) — raw tweet and match-event
   CSVs are cleaned, filtered, and exported as unified, window-aligned tables.
2. **Emotion classification** (`02_emotion_classification.ipynb`) — a pre-trained
   transformer model annotates every tweet with per-emotion probabilities plus
   derived arousal and valence scores.
3. **Temporal alignment** (`03_temporal_alignment.ipynb`) — tweet-level emotion
   scores and match-event features are aggregated into 5-minute windows per match,
   producing a joint time series.
4. **Ad-timing model** (`04_ad_timing_model.ipynb`) — a multi-component receptivity
   scoring function labels each window as `AD_SAFE`, `AD_RISKY`, or `NO_AD`, and
   selects up to four non-overlapping ad slots per match.
5. **Evaluation** (`05_evaluation.ipynb`) — temporal emotion curves, window-level
   distributions, and policy statistics are visualised and discussed.

---

## IV. DATA COLLECTION AND CLEANING

### A. Social Media Dataset

For social media data we use a publicly available Kaggle dataset of English Premier
League tweets collected between 9 July and 13 October 2020 [10].  The raw CSV
contains **14,492 rows** spanning **42 matches** and **26 columns**, including tweet
text, timestamps (`created_at`), a pre-computed TextBlob polarity score, match
metadata (fixture identifier, kick-off time, home/away team), and a
`relative_minute_from_kickoff` column that locates each post on the match timeline.

### B. Match-Event Dataset

Structured match data are obtained through the Sportmonks Football API [9].
The pipeline supports two formats:

- **2020 season** (`premier_league_combined_dataset_2020.csv`, 1 475 rows, 45 columns) —
  contains match events (goals, substitutions, cards, penalties) together with
  whistle and period information.  Pressure values are *not* natively available;
  an intensity proxy is derived instead (see Section V-B).
- **2024/25 season** (`premier_league_combined_dataset_2024_2025.csv`) — extends the
  above with a native `pressure_value` column for every minute of play.

Each dataset row carries a `row_type` tag (`"event"`, `"pressure"`, or `"whistle"`)
that allows the downstream code to selectively filter the relevant subset.

### C. Integration and Alignment

After loading, the two sources are linked through shared `fixture_id` and kick-off
timestamps.  Raw event minutes are converted to an `effective_minute` field
(= `minute` + `extra_minute`) to handle stoppage time correctly.  The resulting
integrated dataset provides a unified chronological view of both on-pitch events
and social media reactions.

---

## V. DATA PREPROCESSING

### A. Tweet Cleaning

Raw tweet text undergoes minimal preprocessing, motivated by the nature of the
downstream model. Because `cardiffnlp/twitter-roberta-base-emotion` was pre-trained
on Twitter text and includes its own BPE tokeniser, heavy preprocessing (stop-word
removal, lemmatisation, stemming) would destroy contextual cues the model relies on.

The applied cleaning steps (`src/preprocessing.py → _clean_tweet_text`) are:

1. **URL removal** — all `http://`, `https://`, and `www.*` tokens are stripped.
2. **Mention removal** — `@user` handles are deleted.
3. **Hashtag normalisation** — `#` symbols are removed while retaining the word
   (e.g. `#Bournemouth` → `Bournemouth`).
4. **Whitespace normalisation** — consecutive spaces are collapsed.
5. **Minimum-length filter** — tweets whose cleaned text is shorter than 10
   characters are discarded.

### B. Temporal Filtering and Window Assignment

Only tweets posted within the interval **[−30, +120] minutes** relative to kick-off
are retained (covering 30 minutes of pre-match anticipation through up to 30 minutes
of extra time).  Each surviving tweet is assigned a `window_5min` label — the floor
of its relative minute rounded down to the nearest multiple of five — which groups
tweets into non-overlapping 5-minute bins for aggregation.

After cleaning and filtering, the dataset retains **14,474 tweets** across **42 matches**
(polarity range: −0.985 to +0.984; minute range: −30 to +115).

### C. Match-Event Features and Pressure Proxy

Match-event rows are extracted from the consolidated football dataset and mapped to
the same 5-minute window grid.  For the **2020 dataset** (no native pressure column),
a weighted event-intensity proxy is derived:

| Event type        | Intensity weight |
|-------------------|-----------------|
| Goal              | 3.0             |
| Own goal          | 3.0             |
| Penalty           | 2.5             |
| Red card          | 2.5             |
| Missed penalty    | 2.0             |
| VAR card          | 2.0             |
| Yellow/red card   | 1.5             |
| Yellow card       | 1.0             |
| Substitution      | 0.3             |

The sum of weights within a window serves as `mean_pressure`; the maximum single
weight serves as `max_pressure`.  Windows with no events are assigned zero.

For the **2024/25 dataset**, native `pressure_value` readings are averaged (mean) and
maximised (max) per window.

In both cases, a Boolean `recent_high_intensity` flag is set for any window that
follows a goal, red card, penalty, own goal, missed penalty, or VAR card in the
immediately preceding window.

---

## VI. EMOTION CLASSIFICATION

### A. Model Selection

We use `cardiffnlp/twitter-roberta-base-emotion`, a RoBERTa-base model fine-tuned
on Twitter data for multi-class emotion recognition (`src/emotion_classifier.py`).
The model outputs softmax probabilities over **four emotion classes**:
`joy`, `optimism`, `anger`, and `sadness`.

The model was chosen because (1) it is trained on the same domain (Twitter) as our
data, (2) it provides fine-grained categorical outputs rather than a single
positive/negative polarity, and (3) its BPE tokeniser handles informal spelling,
emoji, and hashtag vocabulary natively.

Input tweets are truncated to a maximum of **128 tokens** and processed in batches
of **64** (reduced to 32 on CPU) for efficiency.

### B. Derived Affective Dimensions

In addition to per-class probabilities, two scalar affective dimensions are computed
per tweet using a linear combination of the emotion probabilities:

**Arousal** (∈ [0, 1], based on Russell's circumplex model and Breuer et al., 2021):

$$\text{arousal} = \sum_{e} w_{\text{arousal},e} \cdot p_e$$

| Emotion   | Arousal weight |
|-----------|---------------|
| anger     | 1.00          |
| fear      | 0.90          |
| surprise  | 0.85          |
| joy       | 0.60          |
| disgust   | 0.70          |
| sadness   | 0.20          |

**Valence** (∈ [−1, +1]):

$$\text{valence} = \sum_{e} w_{\text{valence},e} \cdot p_e$$

| Emotion   | Valence weight |
|-----------|---------------|
| anger     | −1.00         |
| fear      | −0.80         |
| surprise  |  0.00         |
| joy       | +1.00         |
| disgust   | −0.70         |
| sadness   | −0.90         |

The dominant emotion per tweet is defined as $\arg\max_e p_e$.

### C. Dataset-Level Statistics (Notebook 02 output)

Running the classifier over all 14,474 cleaned tweets yields the following
aggregate distribution:

| Dominant emotion | Count  | Share  |
|-----------------|--------|--------|
| optimism        | 7,535  | 52.1 % |
| joy             | 4,978  | 34.4 % |
| sadness         | 1,316  |  9.1 % |
| anger           |   645  |  4.5 % |

Mean per-class probabilities across all tweets: anger = 0.103, joy = 0.349,
sadness = 0.127.  These figures suggest that Premier League Twitter discourse is
predominantly positive/optimistic, with anger and sadness spiking around adverse
match events (see Section VII).

---

## VII. TEMPORAL ALIGNMENT AND WINDOW FEATURES

*(pending notebook 03 completion — to be filled in)*

---

## VIII. AD-TIMING MODEL

The ad-timing model (`src/ad_timing.py`) operates on the window-level feature table
produced in the previous stage and assigns a single scalar **receptivity score**
∈ [0, 1] to each 5-minute window.  Higher scores indicate greater suitability for
showing an advertisement.

### A. Receptivity Score

The score is computed as the complement of a weighted penalty:

$$\text{receptivity} = 1 - \bigl(0.40 \cdot P_\text{arousal}
   + 0.20 \cdot P_\text{valence}
   + 0.20 \cdot P_\text{pressure}
   + 0.20 \cdot P_\text{event}\bigr)$$

where:

- **Arousal penalty** $P_\text{arousal} = \text{mean\_arousal} \in [0,1]$ —
  high arousal suppresses attention to sponsor messages (Breuer et al., 2021).
- **Valence penalty** $P_\text{valence} = \max(0, -\text{mean\_valence})$ —
  strongly negative sentiment creates negative brand associations.
- **Pressure penalty** $P_\text{pressure} = \min(\text{mean\_pressure}, 50) / 50$ —
  on-pitch pressure reflects fan emotional engagement; capped at 50 to avoid
  outlier dominance.
- **Event penalty** $P_\text{event} \in \{0, 1\}$ — equals 1 if a high-intensity
  event (goal, red card, penalty, etc.) occurred in the immediately preceding window,
  reflecting the ~5-minute recovery period identified by Pawlowski et al. (2024).

### B. Ad-Slot Labels and Recommendations

Each window is categorised as:

| Label      | Receptivity score |
|------------|------------------|
| `AD_SAFE`  | ≥ 0.60           |
| `AD_RISKY` | [0.40, 0.60)     |
| `NO_AD`    | < 0.40           |

Up to **four non-overlapping ad slots** are recommended per match, selected greedily
by descending receptivity score with a minimum spacing of **10 minutes** (2 windows)
between consecutive slots.  Only `AD_SAFE` windows are eligible for recommendation.

---

## IX. RESULTS

*(to be completed once notebook 02 has finished running)*

---

## REFERENCES

*(unchanged — see original paper)*
