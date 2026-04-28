import numpy as np
from scipy.sparse.linalg import svds

def compute_svd(ratings_matrix, k=50):
    user_means = np.mean(ratings_matrix, axis=1)
    R_demeaned = ratings_matrix - user_means.reshape(-1, 1)

    U, sigma, Vt = svds(R_demeaned, k=k)
    sigma = np.diag(sigma)

    predicted = np.dot(np.dot(U, sigma), Vt) + user_means.reshape(-1, 1)
    return predicted