import numpy as np
from scipy.stats import binom
from scipy.optimize import fmin


def compute_sigmoid(array):
    """
    """
    sigmoid = 1/(1 + np.exp(-array))
    return sigmoid


def compute_binom_neg_log_likelihood(p, n=16, k=9):
    """
    """
    negative_log_likelihood = - binom.logpmf(
        k=k,
        n=n,
        p=p
    )
    return negative_log_likelihood


def compute_binom_max_likelihood(start, n=16, k=9):
    """
    """
    solution, iterations = fmin(
        compute_binom_neg_log_likelihood,
        start,
        retall=True
    )
    solution = solution[0]
    iterations = [iter[0] for iter in iterations]
    return solution, iterations


def compute_MC_sampling(frozen_distro, max_sample=1000, iterations=100):
    """
    """
    sampled = {}
    samples = np.linspace(1, max_sample, iterations, dtype=int)
    for sample_size in samples:

        sampled[sample_size] = np.random.choice(frozen_distro, sample_size)

    return sampled


def compute_grid(para_1, para_2):
    """
    """
    grid = []
    for p_1 in para_1:

        for p_2 in para_2:

            grid.append([p_1, p_2])

    grid = np.array(grid)
    return grid


def compute_grid_growth(n_para, n_approx_points):
    """
    """
    grid_growth = [n_approx_points ** power for power in range(1, n_para)]
    return grid_growth


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
    quantiles = np.quantile(sampled_array, [1.0 - ci, ci])
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


def compute_sp_posterior(prior, likelihood):
    """ Compute posterior probability

    Args:
        - prior:
        - likelihood:
    Returns:
        - posterior:
    """
    posterior = prior * likelihood
    posterior = posterior / np.sum(posterior)
    return posterior
