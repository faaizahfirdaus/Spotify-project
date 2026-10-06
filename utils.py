"""
utils.py
--------
Shared preprocessing helpers used by both train_model.py and app.py.
"""

import pandas as pd
import numpy as np

# ── Feature columns used for training ──────────────────────────────────────
FEATURE_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Human-readable labels for the UI sliders
FEATURE_LABELS = {
    "danceability":     ("Danceability",     0.0, 1.0,  0.5),
    "energy":           ("Energy",           0.0, 1.0,  0.5),
    "loudness":         ("Loudness (dB)",   -60.0, 0.0, -10.0),
    "speechiness":      ("Speechiness",      0.0, 1.0,  0.05),
    "acousticness":     ("Acousticness",     0.0, 1.0,  0.3),
    "instrumentalness": ("Instrumentalness", 0.0, 1.0,  0.0),
    "liveness":         ("Liveness",         0.0, 1.0,  0.15),
    "valence":          ("Valence",          0.0, 1.0,  0.5),
    "tempo":            ("Tempo (BPM)",      50.0, 220.0, 120.0),
    "duration_ms":      ("Duration (ms)",    30000.0, 600000.0, 210000.0),
}

POPULARITY_THRESHOLD = 70


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV and return the raw DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and prepare the dataset.

    Steps
    -----
    1. Drop duplicates.
    2. Keep only rows that have all required feature columns.
    3. Drop rows where popularity is missing.
    4. Create binary target: popular = popularity >= POPULARITY_THRESHOLD.
    5. Return feature matrix X and target y.
    """
    df = df.copy()

    # ── 1. Remove exact duplicates ──────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed_dups = before - len(df)

    # ── 2. Normalise column names ───────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Some datasets use "duration_ms", others "duration_ms"; normalise
    if "duration_ms" not in df.columns and "duration" in df.columns:
        df.rename(columns={"duration": "duration_ms"}, inplace=True)

    # ── 3. Retain only required columns + popularity ───────────────────────
    required = FEATURE_COLS + ["popularity"]
    available = [c for c in required if c in df.columns]
    df = df[available].copy()

    # ── 4. Drop rows with any missing values in the required columns ────────
    df.dropna(inplace=True)

    # ── 5. Create binary target ─────────────────────────────────────────────
    y = (df["popularity"] >= POPULARITY_THRESHOLD).astype(int)
    X = df[FEATURE_COLS].copy()

    return X, y


def feature_description(feature_name: str) -> str:
    """Return a one-line description of what a feature measures."""
    descriptions = {
        "danceability":     "How suitable a track is for dancing (rhythm, tempo stability).",
        "energy":           "Perceptual measure of intensity and activity.",
        "loudness":         "Overall loudness in decibels (dB), typically –60 to 0.",
        "speechiness":      "Presence of spoken words in the track.",
        "acousticness":     "Confidence measure of whether the track is acoustic.",
        "instrumentalness": "Predicts whether a track contains no vocals.",
        "liveness":         "Detects the presence of a live audience.",
        "valence":          "Musical positiveness conveyed (happy=high, sad=low).",
        "tempo":            "Overall estimated tempo in beats per minute (BPM).",
        "duration_ms":      "Duration of the track in milliseconds.",
    }
    return descriptions.get(feature_name, "")
