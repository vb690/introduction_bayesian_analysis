import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from modules.stats.backend import compute_sp_posterior

def visualize_grid_search():
    """
    """
    pass


def visualize_sp_update(prior, likelyhood, parameter_space,
                        figsize=(10, 10), **kwargs):
    """ Visualize the prior update.

    Args:
        - prior:
        - likelyhood:

    Returns:
        - None
    """
    prior = np.array(prior)
    likelyhood = np.array(likelyhood)
    posterior = compute_sp_posterior(
        prior=prior,
        likelyhood=likelyhood
    )

    data = {
        'Prior': np.random.choice(
            a=parameter_space,
            size=len(parameter_space),
            p=prior / prior.sum()
        ),
        'Likelyhood': np.random.choice(
            a=parameter_space,
            size=len(parameter_space),
            p=likelyhood / likelyhood.sum()
        ),
        'Posterior': np.random.choice(
            a=parameter_space,
            size=len(parameter_space),
            p=posterior
        ),
    }

    plt.figure(figsize=figsize)
    for name, array in data.items():

        sns.distplot(
            array,
            label=name,
            **kwargs
        )

    plt.xlabel('\u03B8')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def visualize_posterior_predictions():
    """
    """
    pass
