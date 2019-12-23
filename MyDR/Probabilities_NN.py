# 专门为了计算 t-SNE中的联合概率
import numpy as np
from sklearn.manifold import _utils
from time import time
from scipy.sparse import csr_matrix

MACHINE_EPSILON = 1e-12


def _joint_probabilities_nn(distances, neighbors, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : array, shape (n_samples, k)
        Distances of samples to its k nearest neighbors.

    neighbors : array, shape (n_samples, k)
        Indices of the k nearest-neighbors for each samples.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : csr sparse matrix, shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors.
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    n_samples, k = neighbors.shape
    distances = distances.astype(np.float32, copy=False)
    neighbors = neighbors.astype(np.int64, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, neighbors, desired_perplexity, verbose)
    assert np.all(np.isfinite(conditional_P)), \
        "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix((conditional_P.ravel(), neighbors.ravel(),
                    range(0, n_samples * k + 1, k)),
                   shape=(n_samples, n_samples))
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        pass
        # duration = time() - t0
        # print("[t-SNE] Computed conditional probabilities in {:.3f}s"
        #       .format(duration))
    return P

