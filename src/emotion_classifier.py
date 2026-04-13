"""
emotion_classifier.py
---------------------
Wrapper around SamLowe/roberta-base-go_emotions for batch emotion
inference on tweet text.

The model outputs probabilities for 28 emotion classes (GoEmotions):
    admiration, amusement, anger, annoyance, approval, caring,
    confusion, curiosity, desire, disappointment, disapproval,
    disgust, embarrassment, excitement, fear, gratitude, grief,
    joy, love, nervousness, optimism, pride, realization, relief,
    remorse, sadness, surprise, neutral

This model is preferred over cardiffnlp/twitter-roberta-base-emotion
because it distinguishes admiration, excitement, and pride from anger
-- a critical distinction for sports Twitter where emphatic positive
language is frequently misclassified as anger by 4-class models.

Usage
-----
from src.emotion_classifier import EmotionClassifier

clf = EmotionClassifier()
results = clf.predict_batch(tweet_texts)   # list of dicts
"""

from __future__ import annotations

from typing import List, Dict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# Maximum token length
MAX_LENGTH = 128

# Batch size -- reduce if running on CPU
DEFAULT_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Arousal weights (0-1) -- based on Russell's circumplex model.
# High arousal = emotionally activating (good or bad).
# Low arousal  = calm, passive.
# ---------------------------------------------------------------------------
AROUSAL_WEIGHTS = {
    "admiration":     0.50,  # warm but calm
    "amusement":      0.55,  # light, moderate activation
    "anger":          1.00,  # maximum arousal, negative
    "annoyance":      0.65,  # moderate-high arousal, negative
    "approval":       0.40,  # calm positive
    "caring":         0.35,  # warm, low activation
    "confusion":      0.50,  # moderate, uncertain
    "curiosity":      0.50,  # moderate activation
    "desire":         0.60,  # moderate-high
    "disappointment": 0.45,  # moderate, negative
    "disapproval":    0.55,  # moderate-high, negative
    "disgust":        0.70,  # high arousal, negative
    "embarrassment":  0.55,  # moderate
    "excitement":     0.90,  # very high arousal, positive
    "fear":           0.90,  # very high arousal, negative
    "gratitude":      0.40,  # calm positive
    "grief":          0.30,  # low arousal, very negative
    "joy":            0.65,  # high arousal, positive
    "love":           0.50,  # moderate, very positive
    "nervousness":    0.80,  # high arousal, negative
    "optimism":       0.55,  # moderate, positive
    "pride":          0.60,  # moderate-high, positive
    "realization":    0.45,  # moderate
    "relief":         0.30,  # low arousal, positive
    "remorse":        0.35,  # low-moderate, negative
    "sadness":        0.20,  # low arousal, negative
    "surprise":       0.80,  # high arousal, variable valence
    "neutral":        0.30,  # baseline
}

# ---------------------------------------------------------------------------
# Valence weights (+1 = positive, -1 = negative)
# ---------------------------------------------------------------------------
VALENCE_WEIGHTS = {
    "admiration":     0.80,
    "amusement":      0.70,
    "anger":         -1.00,
    "annoyance":     -0.60,
    "approval":       0.70,
    "caring":         0.80,
    "confusion":     -0.20,
    "curiosity":      0.20,
    "desire":         0.50,
    "disappointment":-0.70,
    "disapproval":   -0.65,
    "disgust":       -0.80,
    "embarrassment": -0.50,
    "excitement":     0.90,
    "fear":          -0.80,
    "gratitude":      0.90,
    "grief":         -0.95,
    "joy":            1.00,
    "love":           1.00,
    "nervousness":   -0.60,
    "optimism":       0.70,
    "pride":          0.85,
    "realization":    0.10,
    "relief":         0.80,
    "remorse":       -0.70,
    "sadness":       -0.90,
    "surprise":       0.10,
    "neutral":        0.00,
}

# All 28 GoEmotions labels (canonical order from the dataset)
GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral",
]


class EmotionClassifier:
    """Wrapper around SamLowe/roberta-base-go_emotions (28-class GoEmotions)."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[EmotionClassifier] Loading {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Use model config labels; fall back to canonical list if unavailable
        if hasattr(self.model.config, "id2label"):
            self.labels: List[str] = [
                self.model.config.id2label[i]
                for i in range(self.model.config.num_labels)
            ]
        else:
            self.labels = GO_EMOTIONS_LABELS
        print(f"[EmotionClassifier] {len(self.labels)} labels: {self.labels}")

    # ------------------------------------------------------------------
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict[str, float]]:
        """Run inference on a list of texts.

        The GoEmotions model uses a sigmoid output (multi-label), so we
        apply sigmoid rather than softmax. The dominant emotion is the
        label with the highest probability.

        Returns
        -------
        List of dicts, one per input text, with keys:
            - one key per emotion label (probability, 0-1)
            - 'dominant_emotion'  (str)
            - 'arousal'           (float 0-1)
            - 'valence'           (float -1 to +1)
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits

            # GoEmotions is multi-label: sigmoid per class, not softmax
            probs = torch.sigmoid(logits).cpu().numpy()

            for row in probs:
                result = {label: float(prob) for label, prob in zip(self.labels, row)}
                dominant = self.labels[int(np.argmax(row))]
                result["dominant_emotion"] = dominant

                # Weighted arousal & valence
                # Normalise by sum of probs so scale stays in [0,1] / [-1,1]
                prob_sum = float(np.sum(row)) or 1.0
                result["arousal"] = float(sum(
                    AROUSAL_WEIGHTS.get(lbl, 0.30) * prob
                    for lbl, prob in zip(self.labels, row)
                ) / prob_sum)
                result["valence"] = float(sum(
                    VALENCE_WEIGHTS.get(lbl, 0.00) * prob
                    for lbl, prob in zip(self.labels, row)
                ) / prob_sum)

                all_results.append(result)

        return all_results

    # ------------------------------------------------------------------
    def predict_df(
        self,
        df,
        text_col: str = "text_clean",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Annotate a DataFrame in-place with emotion columns.

        Adds one column per emotion label, plus dominant_emotion,
        arousal, and valence.
        """
        import pandas as pd
        texts = df[text_col].fillna("").tolist()
        results = self.predict_batch(texts, batch_size=batch_size)
        result_df = pd.DataFrame(results)
        for col in result_df.columns:
            df[col] = result_df[col].values
        return df
