# 🎬 Netflix Movie Recommender System

> A personalized movie recommendation engine powered by **Collaborative Filtering (SVD)** — deployed as a fully interactive web app with Streamlit.

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Live Demo

👉 **[Launch the App](https://movie-recommender-system-qxdfea2gstx7jwq4wynccg.streamlit.app/)**
<img width="1904" height="990" alt="image" src="https://github.com/user-attachments/assets/678ec37e-e72b-474e-b09e-c1f0fa3c4c5f" />



---

## 📌 Problem Statement

With millions of titles across Netflix's catalog, users face **decision fatigue** finding content they'll actually enjoy. This project builds a Netflix-style recommender system that learns from a user's historical ratings to deliver **personalized, ranked movie suggestions** — cutting through the noise.

---

## 🧠 How It Works

This system uses **Matrix Factorization via SVD (Singular Value Decomposition)** to uncover hidden patterns in user–movie interaction data.

### Pipeline

```
Raw Ratings Data
      │
      ▼
User-Item Matrix Construction
      │
      ▼
Mean-Centering (remove user bias)
      │
      ▼
SVD Decomposition (SciPy)
      │
      ▼
Rating Matrix Reconstruction
      │
      ▼
Top-N Personalized Recommendations
```

1. **Build a User-Item Matrix** from historical ratings data
2. **Normalize ratings** by subtracting each user's mean (removes individual rating bias)
3. **Apply SVD** to factorize the matrix into latent feature spaces
4. **Reconstruct predictions** by multiplying decomposed matrices
5. **Rank & recommend** unrated movies with the highest predicted scores

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Matrix Factorization | SciPy (`linalg.svds`) |
| Model Evaluation | scikit-learn |
| Web App | Streamlit |
| Poster API | TMDB API (with fallback placeholders) |

---

## ✨ Features

- 🎯 **Personalized recommendations** per user based on their ratings history
- 📊 **Watched movies panel** — shows what the user has already rated
- ⭐ **Predicted rating scores** displayed alongside each recommendation
- 🖼️ **Movie posters** fetched via API, with graceful fallback placeholders
- ⚡ **Memory-efficient** — optimized for large sparse datasets using `float32`
- 🌐 **Fully deployed** — accessible via browser, no setup required

---

## 📊 Model Evaluation

| Metric | Value |
|---|---|
| **RMSE** | ~1.13 |
| Dataset | MovieLens (reduced) |
| Evaluation Strategy | Train/test split on known ratings |

> An RMSE of ~1.13 on a 1–5 rating scale indicates solid predictive accuracy on sparse user-item data. Rankings metrics (Precision@K, Recall@K) are planned for a future release.

---

## 📂 Project Structure

```
Movie-Recommender-System/
│
├── app/
│   └── app.py                # Streamlit UI & user interaction
│
├── src/
│   ├── model.py              # SVD model training & matrix reconstruction
│   ├── recommender.py        # Top-N recommendation generation logic
│   ├── preprocess.py         # Identify movie rows & Rename columns
|   ├── reduce_data.py        # Filtered user-movie ratings
│   └── utils.py              # Poster fetching & helper functions
│
├── data/
│   ├── ratings_reduced.csv   # Filtered user-movie ratings
│   └── movies.csv            # Movie metadata (titles, genres)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nogiayash/Movie-Recommender-System.git

# 2. Navigate into the project
cd Movie-Recommender-System

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ⚡ Performance Optimizations

- **Dataset reduction** — filtered to top users and most-rated movies to keep memory lean
- **`float32` dtype** — halves memory usage vs. default `float64` for the rating matrix
- **Streamlit caching** — `@st.cache_data` prevents redundant SVD recomputation on re-renders
- **Pre-filtering SVD** — truncated SVD applied only to a pre-filtered matrix, avoiding full decomposition cost

---

## ⚠️ Known Limitations

| Limitation | Description |
|---|---|
| Cold Start | New users or movies with no ratings cannot receive recommendations |
| Ratings Only | No content-based signals (genres, cast, description) are used |
| Limited Diversity | SVD tends to recommend popular titles; niche content may be underrepresented |
| Evaluation Scope | Only RMSE is tracked; ranking metrics (Precision@K, Recall@K, NDCG) not yet implemented |

---

## 🔮 Roadmap

- [ ] Add **Precision@K / Recall@K / NDCG** evaluation metrics
- [ ] Build a **Hybrid Recommender** (collaborative + content-based filtering)
- [ ] Integrate **real-time TMDB API** for live poster fetching
- [ ] Redesign UI with a **Netflix-style dark layout**
- [ ] Add a **model retraining pipeline** for fresh data ingestion
- [ ] Implement **ALS (Alternating Least Squares)** as an alternative to SVD

---

## 👨‍💻 Author

**Yashwant Nogia**

- 🐙 GitHub: [@nogiayash](https://github.com/nogiayash)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ using Python & Streamlit</sub>
</div>
