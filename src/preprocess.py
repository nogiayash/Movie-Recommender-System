import zipfile
import pandas as pd

zip_path = "data/ratings_sample.zip"

with zipfile.ZipFile(zip_path) as z:
    file_name = z.namelist()[0]

    with z.open(file_name) as f:
        df = pd.read_csv(
            f,
            header=None,
            names=['Cust_ID', 'Rating'],
            usecols=[0, 1]
        )

# Identify movie rows
df['movieId'] = df['Cust_ID'].where(df['Rating'].isna())

# Forward fill movieId
df['movieId'] = df['movieId'].ffill()

# Clean movieId
df['movieId'] = df['movieId'].astype(str).str.replace(':', '')

# Remove header rows
df = df[df['Rating'].notna()]

# Rename columns
df.rename(columns={'Cust_ID': 'userId', 'Rating': 'rating'}, inplace=True)

# Convert types
df['userId'] = df['userId'].astype(int)
df['movieId'] = df['movieId'].astype(int)
df['rating'] = df['rating'].astype(float)

# 🔥 OPTIONAL (recommended)
df = df.sample(200000)  # limit size for speed

# Save
df.to_csv("data/ratings_sample.csv", index=False)

print("✅ Done (fast)")