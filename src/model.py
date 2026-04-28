import numpy as np
from scipy.sparse.linalg import svds

def compute_svd(ratings_matrix, k=50):

    # Mask of non-zero entries
    mask = ratings_matrix > 0

    # Compute user means ONLY on rated items
    user_means = np.sum(ratings_matrix, axis=1) / np.maximum(mask.sum(axis=1), 1)

    # Demean ONLY rated entries
    R_demeaned = ratings_matrix.copy()
    for i in range(R_demeaned.shape[0]):
        R_demeaned[i, mask[i]] -= user_means[i]

    # SVD
    U, sigma, Vt = svds(R_demeaned, k=k)
    sigma = np.diag(sigma)

    # Reconstruct
    predicted = np.dot(np.dot(U, sigma), Vt)

    # Add means back
    for i in range(predicted.shape[0]):
        predicted[i] += user_means[i]

    return predicted