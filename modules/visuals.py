import numpy as np

from scipy.stats import binom

import matplotlib.pyplot as plt

from IPython.display import display, clear_output

from modules.stats.backend import compute_stats


def visualize_grid_search():
    """
    """
    pass


def visualize_binomial_update(n_tests, parameter_space=np.linspace(0, 1, 100),
                              figsize=(10, 10), **kwargs):
    """ Visualize the prior update.
    """
    remapper = {0: 'Healthy', 1: 'Sick'}
    text_box = '''
        Number of Tests: {}\n
        Number of Positives: {}\n
        MAP: {}\n
        Upper: {}\n
        Lower: {}
    '''

    total_tests_outcomes = 0
    for tests in range(n_tests):

        prob = binom.pmf(
            total_tests_outcomes,
            tests,
            p=parameter_space
        )
        if tests < 1:
            prob = prob / prob.sum()

        stats = compute_stats(
            array=prob,
            ci=95,
            precis=2
        )

        fig = plt.figure(figsize=figsize)
        plt.plot(
            parameter_space,
            prob,
            **kwargs
        )
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel('Plausibility')
        plt.xlabel('Proportion of Sick People')

        plt.text(
            1.1,
            0.9,
            text_box.format(
                tests,
                total_tests_outcomes,
                stats['median'],
                stats['lower'],
                stats['upper']
            ),
            verticalalignment='top'
        )
        if tests > 0:
            plt.title(f'Test Result: {remapper[test_outcome]}')

        test_outcome = np.random.choice([0, 1])
        total_tests_outcomes += test_outcome

        display(fig)
        clear_output(wait=True)
        plt.close()
        plt.pause(0.2)
        if tests < n_tests - 1:
            input()

    return None


def visualize_posterior_predictions():
    """
    """
    pass
