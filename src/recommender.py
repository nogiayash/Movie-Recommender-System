def get_user_recommendations(
    user_id,
    ratings_df,
    predicted_ratings,
    movies_df,
    user_mapper,
    movie_mapper,
    n=10
):

    # -------------------- CHECK USER --------------------
    if user_id not in user_mapper:
        return []

    user_idx = user_mapper[user_id]

    # -------------------- USER PREDICTIONS --------------------
    user_predictions = predicted_ratings[user_idx]

    # Sort movie indices (highest score first)
    sorted_indices = user_predictions.argsort()[::-1]

    # -------------------- WATCHED MOVIES --------------------
    watched_movies = set(
        ratings_df[ratings_df.userId == user_id]["movieId"].values
    )

    # -------------------- FAST LOOKUP --------------------
    reverse_movie_mapper = {v: k for k, v in movie_mapper.items()}

    # Convert movies_df to dict for fast lookup
    movie_dict = dict(zip(movies_df.movieId, movies_df.title))

    # -------------------- NORMALIZATION (OPTIONAL) --------------------
    min_rating = predicted_ratings.min()
    max_rating = predicted_ratings.max()

    recommendations = []

    for idx in sorted_indices:

        if idx not in reverse_movie_mapper:
            continue

        movie_id = reverse_movie_mapper[idx]

        # Skip already watched
        if movie_id in watched_movies:
            continue

        if movie_id not in movie_dict:
            continue

        raw_rating = user_predictions[idx]

        # Scale rating to 1–5 range
        if max_rating != min_rating:
            scaled_rating = 1 + 4 * (raw_rating - min_rating) / (max_rating - min_rating)
        else:
            scaled_rating = raw_rating

        recommendations.append({
            "movieId": movie_id,
            "title": movie_dict[movie_id],
            "rating": float(scaled_rating)
        })

        if len(recommendations) >= n:
            break

    return recommendations