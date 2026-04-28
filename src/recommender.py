def get_user_recommendations(user_id, ratings_df, predicted_ratings, movies_df, user_mapper, movie_mapper, n=10):
    
    # Map userId → row index
    if user_id not in user_mapper:
        return []

    user_row_number = user_mapper[user_id]

    # Get predictions
    user_predictions = predicted_ratings[user_row_number]

    # Sort indices
    sorted_indices = user_predictions.argsort()[::-1]

    # Movies already watched
    watched_movies = set(ratings_df[ratings_df.userId == user_id].movieId.values)

    recommendations = []

    # Reverse movie mapper
    reverse_movie_mapper = {v: k for k, v in movie_mapper.items()}

    for idx in sorted_indices:
        movie_id = reverse_movie_mapper[idx]

        if movie_id in watched_movies:
            continue

        movie_row = movies_df[movies_df.movieId == movie_id]

        if not movie_row.empty:
            recommendations.append({
                "movieId": movie_id,
                "title": movie_row.iloc[0]["title"],
                "rating": float(user_predictions[idx])
            })

        if len(recommendations) >= n:
            break

    return recommendations