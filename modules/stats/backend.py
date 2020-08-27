import numpy as np


def compute_sp_posterior(prior, likelyhood):
    """ Compute posterior probability

    Args:
        - prior:
        - likelyhood:
    Returns:
        - posterior:
    """
    posterior = prior * likelyhood
    posterior = posterior / np.sum(posterior)
    return posterior


def grid_search():
    """
    """
    pass
