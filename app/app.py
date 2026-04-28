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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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
st.write("Personalized movie recommendations")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ratings_reduced.csv")
    movies = pd.read_csv("data/movies.csv")
    return ratings, movies

ratings, movies = load_data()

# -------------------- TRAIN-TEST SPLIT --------------------
@st.cache_data
def split_data(ratings):
    return train_test_split(ratings, test_size=0.2, random_state=42)

train_df, test_df = split_data(ratings)

# -------------------- USER-ITEM MATRIX (TRAIN ONLY) --------------------
train_matrix = train_df.pivot(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

train_matrix = train_matrix.astype('float32')

# -------------------- MAPPINGS --------------------
user_mapper = {user_id: i for i, user_id in enumerate(train_matrix.index)}
movie_mapper = {movie_id: i for i, movie_id in enumerate(train_matrix.columns)}

# -------------------- TRAIN MODEL --------------------
@st.cache_resource
def train_model(matrix):
    return compute_svd(matrix)

predicted_ratings = train_model(train_matrix.values)

# -------------------- RMSE CALCULATION --------------------
def calculate_rmse(test_df, predicted_ratings, user_mapper, movie_mapper):

    actual = []
    predicted = []

    for _, row in test_df.iterrows():
        user = row['userId']
        movie = row['movieId']
        rating = row['rating']

        if user in user_mapper and movie in movie_mapper:
            user_idx = user_mapper[user]
            movie_idx = movie_mapper[movie]

            pred_rating = predicted_ratings[user_idx][movie_idx]

            actual.append(rating)
            predicted.append(pred_rating)

    if len(actual) == 0:
        return None

    return np.sqrt(mean_squared_error(actual, predicted))

rmse = calculate_rmse(test_df, predicted_ratings, user_mapper, movie_mapper)

# -------------------- USER INPUT --------------------
user_id = st.selectbox(
    "Select User ID",
    sorted(train_df['userId'].unique())
)

# -------------------- USER HISTORY --------------------
st.subheader("🎯 Movies You Watched")

user_history = train_df[train_df.userId == user_id].merge(movies, on="movieId")
st.write(user_history[['title', 'rating']].head(5))

# -------------------- RECOMMEND BUTTON --------------------
if st.button("Recommend"):

    with st.spinner("Finding best movies for you..."):
        try:
            recommendations = get_user_recommendations(
                user_id,
                train_df,   # IMPORTANT: use TRAIN data
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

                        # Display scaled rating
                        rating = max(1, min(5, movie['rating']))
                        st.write(f"⭐ {rating:.2f}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# -------------------- MODEL EVALUATION --------------------
    with st.expander("📊 Model Evaluation"):
        if rmse:
            st.write(f"RMSE: {rmse:.3f}")
        else:
            st.write("RMSE could not be computed")