import pandas as pd

df = pd.read_csv("data/ratings_sample.csv")

# Top users
top_users = df['userId'].value_counts().head(1000).index
df = df[df['userId'].isin(top_users)]

# Top movies
top_movies = df['movieId'].value_counts().head(1000).index
df = df[df['movieId'].isin(top_movies)]

df.to_csv("data/ratings_reduced.csv", index=False)

print("✅ Reduced dataset saved")