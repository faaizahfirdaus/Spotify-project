# -*- coding: utf-8 -*-
"""
generate_dataset.py
-------------------
Generates a realistic synthetic Spotify dataset (5,000 songs) and saves it
to data/dataset.csv.

The distributions are modelled after real Spotify audio feature statistics
published in research papers and Kaggle exploratory analyses.

Run:
    python generate_dataset.py
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N    = 5000

rng = np.random.default_rng(SEED)


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Popularity: roughly 15% of songs score >= 70 ────────────────────────
    # Mix two betas: 85% of songs are low-popularity, 15% are high-popularity
    low_pop  = rng.beta(a=2.0, b=5.0, size=n) * 65        # scores 0–65
    high_pop = rng.beta(a=4.0, b=2.0, size=n) * 30 + 70   # scores 70–100
    is_hit   = rng.random(n) < 0.15
    popularity = np.where(is_hit, high_pop, low_pop).astype(int)
    popularity = _clip(popularity, 0, 100)

    popular_mask = popularity >= 70

    # ── Helper: different distributions for popular vs. not-popular songs ───
    def bimodal(pop_mu, pop_sig, npop_mu, npop_sig):
        out = np.where(
            popular_mask,
            rng.normal(pop_mu,  pop_sig,  n),
            rng.normal(npop_mu, npop_sig, n),
        )
        return out

    # ── Audio features ───────────────────────────────────────────────────────
    danceability     = _clip(bimodal(0.68, 0.12, 0.52, 0.16), 0.0, 1.0)
    energy           = _clip(bimodal(0.70, 0.14, 0.55, 0.18), 0.0, 1.0)
    loudness         = _clip(bimodal(-5.5, 2.5, -10.0, 5.0),  -60.0, 0.0)
    speechiness      = _clip(rng.beta(1.5, 12, n),             0.0, 1.0)
    acousticness     = _clip(bimodal(0.18, 0.18, 0.38, 0.28), 0.0, 1.0)
    instrumentalness = _clip(
        np.where(
            rng.random(n) < 0.85,          # 85 % of tracks have vocals
            rng.beta(0.3, 8, n),
            rng.beta(1, 2, n),
        ), 0.0, 1.0
    )
    liveness         = _clip(rng.beta(1.5, 8, n),              0.0, 1.0)
    valence          = _clip(bimodal(0.60, 0.18, 0.45, 0.22), 0.0, 1.0)
    tempo            = _clip(rng.normal(118, 28, n),           50.0, 220.0)
    duration_ms      = _clip(rng.normal(210_000, 55_000, n),   30_000, 600_000).astype(int)

    genres = [
        "pop", "hip-hop", "rock", "r&b", "latin",
        "electronic", "indie", "country", "jazz", "classical",
    ]
    genre_probs = [0.22, 0.17, 0.13, 0.12, 0.10,
                   0.08, 0.07, 0.05, 0.03, 0.03]
    track_genre = rng.choice(genres, n, p=genre_probs)

    artists = [f"Artist_{i}" for i in range(1, 201)]
    artist_name = rng.choice(artists, n)

    track_names = [f"Track_{i:05d}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "track_name":       track_names,
        "artists":          artist_name,
        "track_genre":      track_genre,
        "popularity":       popularity,
        "danceability":     danceability.round(4),
        "energy":           energy.round(4),
        "loudness":         loudness.round(3),
        "speechiness":      speechiness.round(4),
        "acousticness":     acousticness.round(4),
        "instrumentalness": instrumentalness.round(6),
        "liveness":         liveness.round(4),
        "valence":          valence.round(4),
        "tempo":            tempo.round(3),
        "duration_ms":      duration_ms,
    })

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    out_path = os.path.join("data", "dataset.csv")
    df.to_csv(out_path, index=False)
    popular = (df["popularity"] >= 70).sum()
    print(f"Saved {len(df):,} rows to {out_path}")
    print(f"  Popular songs (>= 70) : {popular:,}  ({popular/len(df)*100:.1f}%)")
    print(f"  Not popular           : {len(df)-popular:,}  ({(len(df)-popular)/len(df)*100:.1f}%)")
