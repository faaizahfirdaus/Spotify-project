
# 🎵 Spotify Song Popularity Prediction

A beginner-friendly, end-to-end machine learning project that predicts whether a
Spotify song will be **highly popular** (popularity score ≥ 70) based on its
audio features.

---

## 📋 Project Description

Music labels, artists, and streaming platforms want to know *before* a song is
released whether it has the characteristics of a hit. This project trains a
**Random Forest** (or Logistic Regression) classifier on Spotify audio feature
data to make that prediction.

### Features used for prediction

| Feature | Description | Range |
|---|---|---|
| Danceability | How suitable a track is for dancing | 0 – 1 |
| Energy | Perceptual intensity and activity | 0 – 1 |
| Loudness | Overall loudness in dB | –60 – 0 |
| Speechiness | Presence of spoken words | 0 – 1 |
| Acousticness | Whether the track is acoustic | 0 – 1 |
| Instrumentalness | Absence of vocals | 0 – 1 |
| Liveness | Presence of a live audience | 0 – 1 |
| Valence | Musical positiveness | 0 – 1 |
| Tempo | Beats per minute | 50 – 220 |
| Duration (ms) | Track length in milliseconds | — |

---

## 🗂️ Project Structure

```
spotify_project/
├── app.py            # Streamlit web app (main entry point)
├── train_model.py    # Standalone CLI training script
├── utils.py          # Shared preprocessing helpers
├── requirements.txt  # Python dependencies
├── model/            # Saved model (created after training)
│   └── rf_model.joblib
└── data/             # Put your CSV dataset here
    └── dataset.csv
```

---

## 📦 Dataset

**Recommended:** [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)

1. Visit the link above and download `dataset.csv`.
2. Place it in the `data/` folder:  
   `data/dataset.csv`

The CSV must contain at minimum these columns:
`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`,
`instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `popularity`

---

## ⚙️ Installation

```bash
# 1. Clone / download the project
cd spotify_project

# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — Streamlit Web App (recommended)

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**Steps in the app:**
1. Use the **sidebar** to upload your `dataset.csv`.
2. Click **🚀 Train Model** to train the classifier.
3. Browse the **Model & Evaluation** tab for metrics.
4. Go to the **Predict a Song** tab, adjust sliders, and click **Predict**.

### Option B — Command-line training only

```bash
python train_model.py --data data/dataset.csv
# Logistic Regression instead:
python train_model.py --data data/dataset.csv --model lr
```

The trained model is saved to `model/rf_model.joblib` and loaded
automatically by the Streamlit app on the next run.

---

## 🤖 ML Workflow

```
CSV Upload
    │
    ▼
Preprocessing
  • Remove duplicates
  • Normalise column names
  • Drop rows with missing values
  • Select 10 audio features
  • Create binary target (popularity ≥ 70)
    │
    ▼
Train / Test Split  (80% / 20%, stratified)
    │
    ▼
Model Training
  • Random Forest  (200 trees, max_depth=15)  ← default
  • Logistic Regression  (optional)
    │
    ▼
Evaluation
  • Accuracy, Precision, Recall, F1-score
  • Confusion matrix
  • Feature importance (RF only)
    │
    ▼
Prediction Interface
  • Slider inputs → model → Popular / Not Popular + confidence %
```

---

## 📊 Model Performance (typical, on the Kaggle dataset)

| Metric | Random Forest |
|---|---|
| Accuracy | ~85–88 % |
| Precision | ~80–84 % |
| Recall | ~72–78 % |
| F1-Score | ~76–81 % |

> Exact numbers depend on the dataset split and the class balance.

---

## 🖥️ Demo Interface

The **Predict a Song** tab lets you:

- Drag sliders for each audio feature.
- Click **🎵 Predict Popularity**.
- See:
  - ✅ **Popular** or ❌ **Not Popular**
  - Confidence percentage
  - Probability gauge bar
  - Top-3 most important features driving the prediction

---

## 📝 Notes for Students

- The popularity threshold (70) can be changed in `utils.py → POPULARITY_THRESHOLD`.
- Class imbalance (more *not popular* songs than *popular* ones) is normal and
  expected — the model handles it through the RF's ensemble averaging.
- To try different hyperparameters, edit the `RandomForestClassifier(...)` call
  inside `app.py` or `train_model.py`.
- The project intentionally avoids complex frameworks so you can read and
  understand every line.

---

## 🛠️ Dependencies

| Package | Purpose |
|---|---|
| streamlit | Web UI |
| pandas | Data loading & manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models & evaluation |
| matplotlib | Charts |
| seaborn | Statistical plots |
| joblib | Model serialisation |
