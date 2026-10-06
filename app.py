# -*- coding: utf-8 -*-
"""
app.py
------
Streamlit application for Spotify Song Popularity Prediction.

Sections
--------
1. Sidebar  – dataset upload / sample download link
2. Overview – dataset stats + class distribution chart
3. Train    – train (or retrain) the model; show evaluation metrics
4. Predict  – enter song features → get Popular / Not Popular prediction
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from generate_dataset import generate as generate_sample

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils import (
    load_dataset,
    preprocess,
    FEATURE_COLS,
    FEATURE_LABELS,
    POPULARITY_THRESHOLD,
    feature_description,
)
from train_model import (
    train,
    plot_confusion_matrix,
    plot_feature_importance,
    MODEL_PATH,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Song Popularity Prediction",
    page_icon="🎵",
    layout="wide",
)

# ── Custom CSS (minimal) ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* reduce top padding */
    .block-container { padding-top: 1.5rem; }
    /* metric card */
    [data-testid="metric-container"] {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    /* section headers */
    h2 { border-bottom: 2px solid #3b82d4; padding-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR – Dataset loading
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png",
        width=60,
    )
    st.title("🎵 Spotify Popularity")
    st.markdown("---")

    st.subheader("1. Load Dataset")

    # ── Option A: generate sample data instantly ────────────────────────────
    if st.button("⚡ Use Sample Dataset (5,000 songs)", use_container_width=True,
                 help="Generates a realistic synthetic Spotify dataset — no download needed."):
        with st.spinner("Generating sample dataset …"):
            df_raw = generate_sample(n=5000)
            st.session_state.df = df_raw
            X, y = preprocess(df_raw)
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.pipeline = None
            st.session_state.eval_results = None
        st.success("Sample dataset ready — 5,000 songs generated ✓")

    st.markdown("<div style='text-align:center;color:#888;font-size:12px'>— or —</div>",
                unsafe_allow_html=True)

    # ── Option B: upload your own CSV ───────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload your own Spotify CSV",
        type=["csv"],
        help="Needs columns: danceability, energy, loudness, speechiness, "
             "acousticness, instrumentalness, liveness, valence, tempo, "
             "duration_ms, popularity",
    )
    st.caption(
        "📥 Real dataset: "
        "[Spotify Tracks – Kaggle](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)  \n"
        "or  \n"
        "[Spotify Dataset – Hugging Face](https://huggingface.co/datasets/maharshipandya/-spotify-tracks-dataset)"
    )

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.session_state.df = df_raw
            X, y = preprocess(df_raw)
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.pipeline = None   # force retrain on new data
            st.session_state.eval_results = None
            st.success(f"Loaded {len(df_raw):,} rows ✓")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

    # Auto-load a previously saved model if no dataset uploaded yet
    if st.session_state.pipeline is None and os.path.exists(MODEL_PATH):
        try:
            st.session_state.pipeline = joblib.load(MODEL_PATH)
            st.caption("✅ Pre-trained model loaded from disk.")
        except Exception:
            pass

    st.markdown("---")
    st.subheader("2. Model")
    model_choice = st.radio(
        "Algorithm",
        ["Random Forest", "Logistic Regression"],
        index=0,
        help="Random Forest usually gives better accuracy; "
             "Logistic Regression is faster and more interpretable.",
    )
    model_key = "rf" if model_choice == "Random Forest" else "lr"

    train_btn = st.button(
        "🚀 Train Model",
        disabled=(st.session_state.X is None),
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("Built for a college ML project · uses scikit-learn")


# ── Train on button click ────────────────────────────────────────────────────
if train_btn and st.session_state.X is not None:
    X = st.session_state.X
    y = st.session_state.y

    with st.spinner("Training … this may take a few seconds."):
        # Save a temp CSV so train() can read it back (avoids code duplication)
        os.makedirs("data", exist_ok=True)
        tmp_path = "data/_tmp_training.csv"
        combined = X.copy()
        combined["popularity"] = (y * 100).clip(upper=100)  # dummy raw score
        combined.to_csv(tmp_path, index=False)

        # ── Actually retrain using the in-memory data directly ───────────────
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if model_key == "rf":
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_leaf=4, random_state=42, n_jobs=-1
            )
            pipe = Pipeline([("clf", clf)])
        else:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ])

        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        report  = classification_report(
            y_test, y_pred,
            target_names=["Not Popular", "Popular"],
            output_dict=True,
        )
        cm = confusion_matrix(y_test, y_pred)

        importances = None
        if model_key == "rf":
            importances = dict(zip(FEATURE_COLS,
                                   pipe.named_steps["clf"].feature_importances_))

        os.makedirs("model", exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)

        st.session_state.pipeline = pipe
        st.session_state.eval_results = {
            "accuracy":            acc,
            "report":              report,
            "confusion_matrix":    cm,
            "feature_importances": importances,
            "X_test":              X_test,
            "y_test":              y_test,
            "y_pred":              y_pred,
            "model_name":          model_choice,
        }

    st.success("Model trained and saved ✓")


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════
st.title("🎵 Spotify Song Popularity Prediction")
st.markdown(
    "Predict whether a song is likely to become **highly popular** "
    f"(popularity score ≥ {POPULARITY_THRESHOLD}) based on its audio features."
)

tab_overview, tab_train, tab_predict = st.tabs(
    ["📊 Data Overview", "🤖 Model & Evaluation", "🎯 Predict a Song"]
)


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – DATA OVERVIEW
# ────────────────────────────────────────────────────────────────────────────
with tab_overview:
    if st.session_state.df is None:
        st.info("👈 Upload a dataset in the sidebar to get started.")
        st.markdown(
            """
### About this project

This app trains a **machine learning classifier** to predict whether a
Spotify song will be **highly popular** (popularity score ≥ 70) using
only its audio features.

**Features used:**

| Feature | What it measures |
|---|---|
| Danceability | Suitability for dancing (0–1) |
| Energy | Perceptual intensity and activity (0–1) |
| Loudness | Overall loudness in dB (–60 to 0) |
| Speechiness | Presence of spoken words (0–1) |
| Acousticness | Whether the track is acoustic (0–1) |
| Instrumentalness | Absence of vocals (0–1) |
| Liveness | Presence of live audience (0–1) |
| Valence | Musical positiveness (0–1) |
| Tempo | Beats per minute |
| Duration | Track length in milliseconds |

**ML Workflow:**  
`Upload CSV` → `Preprocess` → `Train / Test Split (80/20)` → `Random Forest` → `Evaluate` → `Predict`

**Dataset source:**  
[Spotify Tracks Dataset – Kaggle](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)
            """
        )
    else:
        df = st.session_state.df
        X  = st.session_state.X
        y  = st.session_state.y

        # ── Summary stats ──────────────────────────────────────────────────
        st.subheader("Dataset Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Features Used", len(FEATURE_COLS))
        c3.metric("Popular Songs", f"{y.sum():,} ({y.mean()*100:.1f}%)")
        c4.metric("Not Popular", f"{(y==0).sum():,} ({(1-y.mean())*100:.1f}%)")

        st.markdown("---")

        # ── Class distribution ─────────────────────────────────────────────
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.subheader("Class Distribution")
            fig1, ax1 = plt.subplots(figsize=(5, 3.5))
            counts = pd.Series({"Not Popular": (y == 0).sum(),
                                 "Popular":    y.sum()})
            colors = ["#93c5fd", "#3b82d4"]
            bars = ax1.bar(counts.index, counts.values, color=colors,
                           edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, counts.values):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + len(y) * 0.005,
                         f"{val:,}", ha="center", va="bottom", fontsize=10)
            ax1.set_ylabel("Number of Songs")
            ax1.set_title("Popular vs. Not Popular", fontweight="bold")
            ax1.spines[["top", "right"]].set_visible(False)
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with col_b:
            st.subheader("Feature Distributions")
            feat_sel = st.selectbox(
                "Select feature to inspect",
                FEATURE_COLS,
                index=0,
            )
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            for label, mask, color in [
                ("Not Popular", y == 0, "#93c5fd"),
                ("Popular",     y == 1, "#3b82d4"),
            ]:
                ax2.hist(X.loc[mask, feat_sel], bins=40, alpha=0.6,
                         color=color, label=label, density=True)
            ax2.set_xlabel(FEATURE_LABELS[feat_sel][0])
            ax2.set_ylabel("Density")
            ax2.set_title(f"Distribution of {FEATURE_LABELS[feat_sel][0]}",
                          fontweight="bold")
            ax2.legend()
            ax2.spines[["top", "right"]].set_visible(False)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # ── Raw data preview ───────────────────────────────────────────────
        with st.expander("🔍 Preview cleaned feature data (first 100 rows)"):
            preview = X.head(100).copy()
            preview["label"] = y.iloc[:100].map(
                {0: "Not Popular", 1: "Popular"}
            ).values
            st.dataframe(preview, use_container_width=True)

        # ── Missing values / preprocessing notes ──────────────────────────
        with st.expander("🧹 Preprocessing Notes"):
            st.markdown(
                f"""
**Steps applied automatically:**

1. **Duplicates removed** – exact duplicate rows dropped.
2. **Column normalisation** – column names lowercased and stripped.
3. **Feature selection** – only the {len(FEATURE_COLS)} audio features listed above are used; metadata like song name, artist, or genre is excluded from training.
4. **Missing values** – rows with any null value in the selected columns are dropped.
5. **Target variable** – `popularity >= {POPULARITY_THRESHOLD}` → **1 (Popular)**, otherwise **0 (Not Popular)**.
6. **Train / test split** – 80 % training, 20 % testing, stratified.
                """
            )


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – MODEL & EVALUATION
# ────────────────────────────────────────────────────────────────────────────
with tab_train:
    if st.session_state.eval_results is None:
        if st.session_state.X is None:
            st.info("Upload a dataset first, then click **Train Model** in the sidebar.")
        else:
            st.info("Click **🚀 Train Model** in the sidebar to train and evaluate the model.")
    else:
        res        = st.session_state.eval_results
        acc        = res["accuracy"]
        report     = res["report"]
        cm         = res["confusion_matrix"]
        importances = res["feature_importances"]
        model_name = res["model_name"]

        st.subheader(f"Evaluation Results – {model_name}")

        # ── Metrics row ────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{acc*100:.2f}%")
        m2.metric("Precision", f"{report['Popular']['precision']*100:.2f}%")
        m3.metric("Recall",    f"{report['Popular']['recall']*100:.2f}%")
        m4.metric("F1-Score",  f"{report['Popular']['f1-score']*100:.2f}%")

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        # ── Confusion matrix ───────────────────────────────────────────────
        with col1:
            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix(cm)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            with st.expander("How to read the confusion matrix"):
                st.markdown(
                    """
- **True Negative (top-left):** Correctly predicted *Not Popular*.
- **False Positive (top-right):** Wrongly predicted *Popular* when it was *Not Popular*.
- **False Negative (bottom-left):** Wrongly predicted *Not Popular* when it was *Popular*.
- **True Positive (bottom-right):** Correctly predicted *Popular*.
                    """
                )

        # ── Feature importance ─────────────────────────────────────────────
        with col2:
            if importances:
                st.subheader("Feature Importance")
                fig_fi = plot_feature_importance(importances)
                st.pyplot(fig_fi)
                plt.close(fig_fi)

                top_feat = max(importances, key=importances.get)
                st.info(
                    f"🏆 Most important feature: **{FEATURE_LABELS[top_feat][0]}**  \n"
                    f"{feature_description(top_feat)}"
                )
            else:
                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).T.round(3)
                st.dataframe(report_df, use_container_width=True)

        # ── Full classification report ─────────────────────────────────────
        st.markdown("---")
        st.subheader("Full Classification Report")
        report_df = pd.DataFrame(report).T.iloc[:-3].round(4)
        report_df.index.name = "Class"
        st.dataframe(report_df, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 – PREDICT A SONG
# ────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("🎯 Predict Song Popularity")

    if st.session_state.pipeline is None:
        st.warning(
            "No trained model found.  \n"
            "Upload a dataset and click **🚀 Train Model** in the sidebar, "
            "or use the pre-trained model if one exists on disk."
        )
    else:
        pipe = st.session_state.pipeline

        st.markdown(
            "Adjust the sliders below to match your song's audio features, "
            "then click **Predict**."
        )

        # ── Input sliders ──────────────────────────────────────────────────
        col_l, col_r = st.columns(2)

        inputs = {}
        slider_order = [
            ("danceability", col_l),
            ("energy",       col_r),
            ("loudness",     col_l),
            ("speechiness",  col_r),
            ("acousticness", col_l),
            ("instrumentalness", col_r),
            ("liveness",     col_l),
            ("valence",      col_r),
            ("tempo",        col_l),
            ("duration_ms",  col_r),
        ]

        for feat, col in slider_order:
            label, lo, hi, default = FEATURE_LABELS[feat]
            with col:
                inputs[feat] = st.slider(
                    label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=(hi - lo) / 200,
                    help=feature_description(feat),
                )

        predict_btn = st.button("🎵 Predict Popularity", type="primary",
                                use_container_width=True)

        if predict_btn:
            input_df = pd.DataFrame([inputs])[FEATURE_COLS]
            prediction = pipe.predict(input_df)[0]
            proba      = pipe.predict_proba(input_df)[0]
            confidence = proba[prediction]

            st.markdown("---")

            if prediction == 1:
                st.success(
                    f"### 🌟 Popular\n\n"
                    f"This song is predicted to be **highly popular**  \n"
                    f"Confidence: **{confidence*100:.1f}%**"
                )
            else:
                st.error(
                    f"### 📉 Not Popular\n\n"
                    f"This song is predicted to be **not highly popular**  \n"
                    f"Confidence: **{confidence*100:.1f}%**"
                )

            # ── Probability gauge ──────────────────────────────────────────
            fig_p, ax_p = plt.subplots(figsize=(6, 1.2))
            pop_prob = proba[1]
            ax_p.barh(["Popularity\nProbability"], [pop_prob],
                      color="#3b82d4", height=0.5)
            ax_p.barh(["Popularity\nProbability"], [1 - pop_prob],
                      left=[pop_prob], color="#e5e7eb", height=0.5)
            ax_p.axvline(0.5, color="gray", linestyle="--", linewidth=1)
            ax_p.set_xlim(0, 1)
            ax_p.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_p.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax_p.spines[["top", "right", "left"]].set_visible(False)
            ax_p.set_title(
                f"Predicted probability of being Popular: {pop_prob*100:.1f}%",
                fontsize=11
            )
            fig_p.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

            # ── Feature influence explanation ──────────────────────────────
            if st.session_state.eval_results and \
               st.session_state.eval_results.get("feature_importances"):
                importances = st.session_state.eval_results["feature_importances"]
                top3 = sorted(importances.items(), key=lambda x: x[1],
                               reverse=True)[:3]

                st.markdown("#### 💡 Key features influencing this prediction")
                for rank, (feat, imp) in enumerate(top3, 1):
                    label = FEATURE_LABELS[feat][0]
                    val   = inputs[feat]
                    desc  = feature_description(feat)
                    st.markdown(
                        f"**{rank}. {label}** — value: `{val:.3f}` "
                        f"(importance: {imp*100:.1f}%)  \n"
                        f"<small>{desc}</small>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "Train a **Random Forest** model to see feature importances."
                )
