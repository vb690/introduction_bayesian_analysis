import numpy as np

from scipy.stats import binom

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, clear_output

from modules.stats.backend import compute_stats, compute_sp_posterior


def visualize_priors_effect(parameter_space, priors, likelyhood,
                            figsize=(15, 10), **kwargs):
    """
    """
    sns.set(style='white', font_scale=1.5)
    fig, axs = plt.subplots(1, len(priors), figsize=figsize)
    for prior_key, ax in zip(priors, axs.flatten()):

            posterior = compute_sp_posterior(
                prior=priors[prior_key],
                likelyhood=likelyhood,
            )
            outcomes = {
                'Prior': priors[prior_key],
                'Likelyhood': likelyhood,
                'Posterior': posterior,
            }

            for outcome_name, outcome in outcomes.items():

                ax.plot(
                    parameter_space,
                    outcome / outcome.sum(),
                    label=outcome_name
                )

            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_title(prior_key)
            ax.set_ylabel('Plausibility')
            ax.set_xlabel('Proportion of Sick People')
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.03),
        ncol=3
    )
    plt.tight_layout()
    plt.show()

def visualize_binomial_update(n_tests, parameter_space=np.linspace(0, 1, 100),
                              figsize=(10, 10), auto=False, **kwargs):
    """ Visualize the prior update.
    """
    sns.set(style='white', font_scale=1.5)
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
            array=parameter_space,
            ci=0.95,
            precis=3,
            sampling_prob=prob,
        )

        fig = plt.figure(figsize=figsize)
        plt.plot(
            parameter_space,
            prob,
            **kwargs
        )
        plt.yticks([])
        plt.xlim(0, 1)
        plt.ylabel('Plausibility')
        plt.xlabel('Proportion of Sick People')

        plt.text(
            1.05,
            0.9,
            text_box.format(
                tests,
                total_tests_outcomes,
                stats['median'],
                stats['upper'],
                stats['lower']
            ),
            verticalalignment='top',
            transform=fig.transFigure
        )
        if tests > 0:
            plt.title(f'Test Result: {remapper[test_outcome]}')

        test_outcome = np.random.choice([0, 1])
        total_tests_outcomes += test_outcome

        display(fig)
        clear_output(wait=True)
        plt.close()
        plt.pause(1 / n_tests)
        if tests < n_tests - 1 and not auto:
            input()

    return None

def visualize_bivariate_regression(X, y, X_label='', y_label='', title='',
                                   figsize=(10, 8), **kwargs):
    sns.set(style='white', font_scale=1.5)
    plt.figure(figsize=(10, 8))
    plt.scatter(X, y)

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(X_label)
    plt.show()

    return None

def visualize_posterior_predictions():
    """
    """
    pass
