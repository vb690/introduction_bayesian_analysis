import numpy as np


def compute_stats(array, ci=95, precis=2):
    """
    """
    array = np.array(array)
    stats = {
        'mean': array.mean(),
        'std': array.std(),
        'median': np.median(array),
        'lower': np.percentile(array, 100 - ci),
        'upper': np.percentile(array, ci)
    }
    for stat, value in stats.items():

        stats[stat] = np.round(value, precis)

    return stats


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
