import sys
import os
import streamlit as st
import pandas as pd
import numpy as np


# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import compute_svd
from recommender import get_user_recommendations
from utils import fetch_poster
from evaluation import compute_rmse



# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# -------------------- STYLING --------------------
st.markdown("""
<style>
img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🎬 Netflix Movie Recommender System")
st.write("Personalized movie recommendations ")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ratings_sample.csv")
    movies = pd.read_csv("data/movies.csv")
    return ratings, movies

ratings, movies = load_data()

# -------------------- REDUCE DATA SIZE (CRITICAL FIX) --------------------
@st.cache_data
def reduce_data(ratings):
    # Top active users
    top_users = ratings['userId'].value_counts().head(1000).index
    ratings = ratings[ratings['userId'].isin(top_users)]

    # Top popular movies
    top_movies = ratings['movieId'].value_counts().head(1000).index
    ratings = ratings[ratings['movieId'].isin(top_movies)]

    return ratings

ratings = reduce_data(ratings)

# -------------------- USER-ITEM MATRIX --------------------
user_item_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# 🔥 Reduce memory
user_item_matrix = user_item_matrix.astype('float32')

user_mapper = {user_id: i for i, user_id in enumerate(user_item_matrix.index)}
movie_mapper = {movie_id: i for i, movie_id in enumerate(user_item_matrix.columns)}

# -------------------- TRAIN MODEL --------------------
@st.cache_resource
def train_model(matrix):
    return compute_svd(matrix)

predicted_ratings = train_model(user_item_matrix.values)

# -------------------- USER INPUT --------------------
max_user_id = int(ratings.userId.max())

user_id = st.selectbox(
    "Select User ID",
    sorted(ratings['userId'].unique())
)

# -------------------- User History --------------------

st.subheader("🎯 Movies You Watched")

user_history = ratings[ratings.userId == user_id].merge(movies, on="movieId")

st.write(user_history[['title', 'rating']].head(5))

# -------------------- RECOMMEND BUTTON --------------------
if st.button("Recommend"):
    if user_id not in ratings.userId.values:
        st.error("User ID not found in filtered dataset")
    else:
        with st.spinner("Finding best movies for you..."):
            try:
                recommendations = get_user_recommendations(
                    user_id,
                    ratings,
                    predicted_ratings,
                    movies,
                    user_mapper,
                    movie_mapper,
                    n=10
                )

                if not recommendations:
                    st.warning("No recommendations found")
                else:
                    st.subheader("Top Recommendations")

                    cols = st.columns(5)

                    for i, movie in enumerate(recommendations):
                        with cols[i % 5]:
                            poster = fetch_poster(movie["title"])
                            st.image(poster, use_container_width=True)
                            st.markdown(f"**{movie['title']}**")
                            st.write(f"⭐ {movie['rating']:.2f}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    # -------------------- MODEL EVALUATION --------------------
    with st.expander("📊 Model Evaluation"):
        RMSE = compute_rmse(user_item_matrix.values, predicted_ratings)
        st.write(f"RMSE: {RMSE}")
        st.write("Metric used to evaluate prediction accuracy")