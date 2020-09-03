import numpy as np


def compute_stats(array, ci=0.95, precis=2, sampling_prob=None):
    """
    """
    if sampling_prob is not None:
        sampled_array = np.random.choice(
            array,
            size=len(array),
            p=sampling_prob / sampling_prob.sum()
        )
    else:
        sampled_array = np.array(array)
    quantiles =  np.quantile(sampled_array, [1.0 - ci, ci])
    stats = {
        'mean': sampled_array.mean(),
        'std': sampled_array.std(),
        'median': np.median(sampled_array),
        'lower': min(quantiles),
        'upper': max(quantiles)
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
