"""
emotion_classifier.py
---------------------
Wrapper around cardiffnlp/twitter-roberta-base-emotion for batch
emotion inference on tweet text.

The model outputs probabilities for 4 emotions:
    anger, fear, joy, sadness

NOTE: some checkpoint versions of this model also output 'optimism'.
We include weights for all observed labels. Labels not present in a
given checkpoint are simply never matched (the get() call below
defaults gracefully), so no code change is needed if the label set
varies between checkpoints.

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

MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"

# Maximum token length (model limit is 514; we truncate conservatively)
MAX_LENGTH = 128

# Batch size — reduce if running on CPU
DEFAULT_BATCH_SIZE = 32

# Arousal mapping: how much each emotion contributes to viewer arousal.
# Based on Russell's circumplex model + Breuer et al. (2021) findings.
# 'optimism' added because some Cardiff checkpoint versions output it;
# mapped to moderate-high arousal / positive valence (similar to joy).
AROUSAL_WEIGHTS = {
    "anger":     1.0,   # high arousal, negative valence
    "fear":      0.9,   # high arousal, negative valence
    "joy":       0.6,   # moderate-high arousal, positive valence
    "sadness":   0.2,   # low arousal, negative valence
    "optimism":  0.55,  # moderate arousal, positive valence
}

# Valence mapping (+1 positive, -1 negative)
VALENCE_WEIGHTS = {
    "anger":    -1.0,
    "fear":     -0.8,
    "joy":       1.0,
    "sadness":  -0.9,
    "optimism":  0.6,
}


class EmotionClassifier:
    """Thin wrapper around the Cardiff NLP twitter-roberta-base-emotion model."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[EmotionClassifier] Loading {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Extract label list from model config
        self.labels: List[str] = [
            self.model.config.id2label[i]
            for i in range(self.model.config.num_labels)
        ]
        print(f"[EmotionClassifier] Labels: {self.labels}")

    # ------------------------------------------------------------------
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict[str, float]]:
        """Run inference on a list of texts.

        Returns
        -------
        List of dicts, one per input text, with keys:
            - one key per emotion label (probability)
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
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                result = {label: float(prob) for label, prob in zip(self.labels, row)}
                dominant = self.labels[int(np.argmax(row))]
                result["dominant_emotion"] = dominant

                # Weighted arousal & valence — unknown labels default to
                # neutral (0.5 arousal, 0.0 valence) via get()
                result["arousal"] = float(sum(
                    AROUSAL_WEIGHTS.get(lbl, 0.5) * prob
                    for lbl, prob in zip(self.labels, row)
                ))
                result["valence"] = float(sum(
                    VALENCE_WEIGHTS.get(lbl, 0.0) * prob
                    for lbl, prob in zip(self.labels, row)
                ))

                all_results.append(result)

        return all_results

    # ------------------------------------------------------------------
    def predict_df(self, df, text_col: str = "text_clean", batch_size: int = DEFAULT_BATCH_SIZE):
        """Annotate a DataFrame in-place with emotion columns.

        Adds columns: anger, fear, joy, sadness (+ any model labels),
        dominant_emotion, arousal, valence.
        """
        texts = df[text_col].fillna("").tolist()
        results = self.predict_batch(texts, batch_size=batch_size)
        result_df = __import__("pandas").DataFrame(results)
        for col in result_df.columns:
            df[col] = result_df[col].values
        return df
