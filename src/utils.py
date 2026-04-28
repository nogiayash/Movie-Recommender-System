import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

def fetch_poster(movie_name):
    return f"https://via.placeholder.com/300x450?text={movie_name.replace(' ', '+')}"