# TMDB Movie Success Prediction

An explainable machine learning pipeline that predicts a movie's box-office success **before it is released**, built on the TMDB 5000 movie dataset. The project engineers predictive features from raw movie metadata (cast, crew, genres, keywords, taglines), evaluates multiple ML models, and uses SHAP to explain *why* the model predicts what it predicts — answering 12 concrete business research questions along the way.

## What this project does

- Cleans and merges the raw TMDB movies + credits datasets (JSON-in-string parsing, missing-value handling, deduplication)
- Runs a 13-part exploratory data analysis answering questions like "does release timing matter?", "do directors/actors move the needle?", "what makes a movie flop?"
- Engineers ~60 features (financial, seasonal, genre, keyword/theme, director & actor target-encoding, studio, runtime, tagline)
- Trains and compares Linear/Logistic Regression, Random Forest, XGBoost, and LightGBM for both **revenue regression** and **hit/flop classification**
- Tunes hyperparameters with `RandomizedSearchCV` and checks for overfitting via learning curves
- Uses **SHAP** to explain the tuned models and answer all 12 research questions with evidence
- Validates the pre-release model against 10 real 2023–2024 releases (Barbie, Oppenheimer, Dune: Part Two, etc.)

### Leakage-free by design: Track A vs. Track B

A naive model would use `vote_count` and `popularity` — but those only exist *after* a movie is already in theaters, which is a classic data leakage trap for a "predict before release" use case. This project explicitly splits modeling into two tracks:

| Track | Features | Scenario | Best result |
|-------|----------|----------|-------------|
| **A — Full model** | Includes post-release signals (`vote_count`, `popularity`, `vote_average`) | Retrospective analysis / streaming recommendation | R² = 0.38 (regression), AUC = 0.936 (classification) |
| **B — Pre-release model** | Only features knowable before a movie comes out (budget, genre, cast/director history, studio, release date) | Greenlight / pre-release decision support | AUC = 0.884 (classification) |

Track B is the one that matters for real decision-making: it proves a movie's hit probability can be estimated from budget, studio backing, director/cast track record, and release timing alone — with no knowledge of how audiences will actually respond.

## Key findings

- **Budget is the strongest pre-release predictor** (SHAP = 0.64 in Track B), followed by major-studio backing and director/actor track record
- **Top directors (top 50 by revenue) have a 15.5× revenue gap and a 92% hit rate** vs. the rest of the dataset
- **A June release earns a 4× higher median revenue than September** — the classic summer-blockbuster vs. "dump month" effect
- **A high critic rating does not guarantee financial success** — Transformers: Age of Extinction (5.8/10) grossed $1.09B
- **150–180 minute runtime is the revenue sweet spot**; beyond that, returns drop off
- Full write-up of all 12 research questions with supporting evidence is in `notebooks/tmdb_movie_analysis.ipynb`, Step 6.13

## Project structure

```
tmdb-movie-project/
├── data/                  # Raw, cleaned, and feature-engineered datasets + train/test splits
├── models/                # Trained model artifacts (.pkl) and fitted scalers
├── notebooks/
│   └── tmdb_movie_analysis.ipynb   # Full pipeline: cleaning → EDA → features → modeling → SHAP insights
├── reports/                # Generated charts (SHAP, feature importance, EDA) and comparison tables
└── requirements.txt
```

## Pipeline

| Step | What happens |
|------|---------------|
| 0 — Data Cleaning | Parse JSON-string columns, merge movies + credits, handle missing values, dedupe |
| 1 — EDA | 13 analyses covering revenue/budget, genre, seasonality, director/actor effect, language, studio, runtime, decade trends, keywords, flop risk |
| 2 — Feature Engineering | ~60 engineered features derived directly from EDA findings |
| 3 — Preprocessing | Track A / Track B split, train/test split, scaling, encoding |
| 4 — Model Building | Baselines → Linear/Logistic → Random Forest → XGBoost → LightGBM, compared side by side |
| 5 — Evaluation & Tuning | Detailed regression/classification diagnostics, `RandomizedSearchCV` tuning, learning curves |
| 6 — Inference & Insights | SHAP-based answers to all 12 research questions, single-movie prediction function |
| 7 — Track B Deep Dive | Dedicated pre-release-only modeling track + validation on 2023–2024 releases |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/tmdb_movie_analysis.ipynb
```

The notebook is fully self-contained and re-runnable top to bottom — it reads from `data/`, writes trained models to `models/`, and writes charts/tables to `reports/`.

To predict a new movie, edit the `film` dictionary in the "Single-Movie Prediction" cell near the end of the notebook (Step 6) and run it — it only uses features knowable before release.

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (`tmdb_5000_movies.csv` + `tmdb_5000_credits.csv`), sourced from Kaggle / The Movie Database (TMDB).

## Roadmap

- [ ] Interactive Looker Studio dashboard presenting the model's business insights to non-technical stakeholders

## Limitations

- Revenue regression shows a noticeable train/validation gap (learning curves, Step 5.7) — revenue is inherently high-variance and influenced by factors (marketing spend, cultural moment, social media buzz) not present in this dataset
- The dataset skews toward English-language (96%) and 2000s–2010s releases (91%), which limits generalization to other markets and eras
- Train/test split is random rather than temporal; a chronological split would give a more realistic estimate of forecasting future releases
