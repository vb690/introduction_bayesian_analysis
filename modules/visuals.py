import numpy as np

from scipy.stats import binom

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, clear_output

from modules.stats.backend import compute_sigmoid
from modules.stats.backend import compute_grid, compute_grid_growth
from modules.stats.backend import compute_binom_max_likelihood
from modules.stats.backend import compute_MC_sampling
from modules.stats.backend import compute_stats, compute_sp_posterior


def visualize_grid_search(n_para, n_approx_points=100,
                          figsize=(15, 8), **kwargs):
    """
    """
    grid = compute_grid(
        para_1=np.linspace(0, 1, n_approx_points),
        para_2=np.linspace(0, 1, n_approx_points)
    )

    grid_growth = compute_grid_growth(
        n_para=n_para,
        n_approx_points=n_approx_points
    )

    sns.set(style='white', font_scale=1.5)
    fig, axs = plt.subplots(1, 3, figsize=figsize, **kwargs)

    axs[0].scatter(
        np.linspace(0, 1, n_approx_points),
        [0.5] * n_approx_points,
        s=0.5,
        c='r'
    )
    axs[0].set_title('Single Parameter Grid')
    axs[0].set_xlabel('$\\theta$')
    axs[0].set_yticks([])

    axs[1].scatter(
        grid[:, 0],
        grid[:, 1],
        0.5,
        c='r'
    )
    axs[1].set_title('Two Parameters Grid')
    axs[1].set_xlabel('$\\theta 1$')
    axs[1].set_ylabel('$\\theta 2$')

    axs[2].plot(
        [i for i in range(1, n_para)],
        [np.log10(float(n_points)) for n_points in grid_growth],
        c='r'
    )
    axs[2].set_title('Grid Growth')
    axs[2].set_xticks([i for i in range(1, n_para)])
    axs[2].set_xlabel('Number of Parameters')
    axs[2].set_ylabel('$log10$(Points in the Grid)')

    plt.tight_layout()
    plt.show()

    return None


def visulize_quadratic_approx(parameter_space, n, k, figsize=(10, 8),
                              **kwargs):
    """
    """
    plausibility = binom.pmf(
        k=k,
        n=n,
        p=parameter_space
    )
    sns.set(style='white', font_scale=1.5)
    fig = plt.figure(figsize=figsize)
    plt.plot(
        parameter_space,
        plausibility,
        **kwargs
    )
    plt.xlabel('$\\theta$')
    plt.yticks([])
    plt.show()
    input()
    solution, iterations = compute_binom_max_likelihood(
        0.0,
        n=n,
        k=k

    )
    for iteration in iterations:

        likelihood = binom.pmf(
            k=k,
            n=n,
            p=iteration
        )

        fig = plt.figure(figsize=figsize)
        plt.plot(
            parameter_space,
            plausibility,
            **kwargs
        )
        plt.scatter(
            iteration,
            likelihood,
            c='r'
        )
        circle = plt.Circle(
            (iteration, likelihood),
            0.01,
            color='r',
            fill=False
        )
        plt.gcf().gca().add_artist(circle)
        plt.xlabel('$\\theta$')
        plt.yticks([])
        plt.title(
            f'$\\theta$ {round(iteration, 3)} Likely: {round(likelihood, 3)}'
        )

        display(fig)
        clear_output(wait=True)
        plt.close()
        plt.pause(0.5)


def visualize_MC_sampling(frozen_distro, max_sample=1000, iterations=100,
                          figsize=(10, 8), **kwargs):
    """
    """
    sns.set(style='white', font_scale=1.5)
    sampled = compute_MC_sampling(
        frozen_distro=frozen_distro,
        max_sample=max_sample,
        iterations=iterations
    )
    fig = plt.figure(figsize=figsize)
    sns.distplot(
        frozen_distro,
        kde=True,
        hist=False,
        color='r'
    )
    plt.xlabel('$\\theta$')
    plt.yticks([])
    plt.show()
    input()
    for sample_size, sample in sampled.items():

        fig = plt.figure(figsize=figsize)
        sns.distplot(
            frozen_distro,
            kde=True,
            hist=False,
            color='r'
        )
        sns.distplot(
            sample,
            kde=True,
            hist=False,
            color='b',
            **kwargs
        )

        plt.xlabel('$\\theta$')
        plt.yticks([])
        plt.title(
            f'Samples: {sample_size}'
        )
        display(fig)
        clear_output(wait=True)
        plt.close()
        plt.pause(0.5)


def visualize_priors_effect(parameter_space, priors, likelihood,
                            figsize=(10, 8), **kwargs):
    """
    """
    sns.set(style='white', font_scale=1.5)
    fig, axs = plt.subplots(1, len(priors), figsize=figsize)
    for prior_key, ax in zip(priors, axs.flatten()):

        posterior = compute_sp_posterior(
                prior=priors[prior_key],
                likelihood=likelihood,
            )
        outcomes = {
            'Prior': priors[prior_key],
            'likelihood': likelihood,
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
            ax.set_xlabel('Proportion')
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

    return None


def visualize_binomial_update(n_tests, parameter_name, remapper,
                              outcome_p=[0.5, 0.5],
                              parameter_space=np.linspace(0, 1, 100),
                              hist=True, figsize=(10, 8),
                              auto=False, **kwargs):
    """ Visualize the prior update.
    """
    sns.set(style='white', font_scale=1.5)
    text_box = '''
        Number Total: {}\n
        Number {}: {}\n
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
        if hist:
            sns.barplot(
                parameter_space,
                prob,
                **kwargs
            )
        else:
            plt.plot(
                parameter_space,
                prob,
                **kwargs
            )
            plt.xticks(parameter_space)
        plt.yticks([])
        plt.ylabel('Plausibility')
        plt.xlabel(f'Proportion of {parameter_name}')

        plt.text(
            1.05,
            0.9,
            text_box.format(
                tests,
                parameter_name,
                total_tests_outcomes,
                stats['median'],
                stats['upper'],
                stats['lower']
            ),
            verticalalignment='top',
            transform=fig.transFigure
        )
        if tests > 0:
            plt.title(f'Observation: {remapper[test_outcome]}')

        test_outcome = np.random.choice(
            [0, 1],
            p=outcome_p
        )
        total_tests_outcomes += test_outcome

        display(fig)
        clear_output(wait=True)
        plt.close()
        plt.pause(1 / n_tests)
        if tests < n_tests - 1 and not auto:
            input()

    return None


def visualize_bivariate_relationship(X, y, X_label='', y_label='', title='',
                                   figsize=(10, 8), **kwargs):
    """
    """
    sns.set(style='white', font_scale=1.5)
    plt.figure(figsize=figsize)
    plt.scatter(X, y, **kwargs)

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(X_label)
    plt.show()

    return None

def visualize_time_series(X, y, prediction_point, X_label, y_label,
                          prediction=None, figsize=(8, 8)):
    """
    """
    sns.set(style='white', font_scale=1.5)
    plt.figure(figsize=figsize)

    plt.plot(
        X[:prediction_point],
        y[:prediction_point],
        c='b'
    )
    plt.plot(
        X[prediction_point:],
        y[prediction_point:],
        alpha=0.5,
        c='b',
        linestyle='--'
    )

    if prediction is not None:
        plt.plot(
            X[prediction_point:],
            prediction.mean(axis=0),
            c='b'
        )
        plt.plot(
            X[prediction_point:],
            np.percentile(prediction, 5, axis=0),
            alpha=0.5,
            c='r',
            linestyle='--'
        )
        plt.plot(
            X[prediction_point:],
            np.percentile(prediction, 90, axis=0),
            alpha=0.5,
            c='r',
            linestyle='--'
        )
    plt.axvline(prediction_point, c='r', linestyle='--')
    plt.ylabel(y_label)
    plt.xlabel(X_label)
    plt.show()

    return None


def visualize_regression_lines(X, y, intercepts, slopes, title,
                               figsize=(10, 8), overlay=True, predictions=None,
                               logistic=False, **kwargs):
    """
    """
    alpha_1 = 0.3 if overlay else 1
    alpha_2 = 1 if overlay else 0.3

    sort_index = X.argsort()
    X = X[sort_index]
    y = y[sort_index]

    sns.set(style='white', font_scale=1.5)
    plt.figure(figsize=figsize)

    predictor = np.linspace(X.min(), X.max(), len(X))

    if predictions is not None:
        predictions = predictions[sort_index]
        lines = intercepts.reshape(-1, 1) + \
                slopes.reshape(-1, 1) * predictor.reshape(1, -1)
        if logistic:
            lines = np.apply_along_axis(
                compute_sigmoid,
                axis=0,
                arr=lines
            )

        lines_mean = lines.mean(axis=0)
        lines_percentiles = np.percentile(lines, [5, 90], axis=0)
        predictions_percentiles = np.percentile(predictions, [5, 90], axis=0)

        plt.plot(
            predictor,
            lines_mean,
            alpha=alpha_1,
            c='r'
        )
        plt.plot(
            predictor,
            lines_percentiles[0, :],
            alpha=alpha_1 * 0.75,
            linestyle='dotted',
            color='r'
        )
        plt.plot(
            predictor,
            lines_percentiles[1, :],
            alpha=alpha_1 * 0.75,
            linestyle='dotted',
            color='r'
        )
        if not logistic:
            plt.plot(
                [X.min(), X.max()],
                [
                    predictions_percentiles[0, :].min(),
                    predictions_percentiles[0, :].max()
                ],
                alpha=alpha_1 * 0.75,
                linestyle='dashed',
                color='b'
            )
            plt.plot(
                [X.min(), X.max()],
                [
                    predictions_percentiles[1, :].min(),
                    predictions_percentiles[1, :].max()
                ],
                alpha=alpha_1 * 0.75,
                linestyle='dashed',
                color='b'
            )
    else:
        for intercept, slope in zip(intercepts, slopes):

            line = intercept + slope * predictor
            if logistic:
                line = compute_sigmoid(line)
            plt.plot(
                predictor,
                line,
                alpha=alpha_1,
                **kwargs
            )

    plt.scatter(
        X,
        y,
        alpha=alpha_2,
        c='b'
    )
    plt.title(title)
    plt.xlabel('Predictor X')
    plt.ylabel('Outcome y')
    plt.show()

    return None


def visualize_bivariate_parameter_grid(parameter_1, parameter_2,
                                       parameter_1_name, parameter_2_name,
                                       height=10):
    """
    """
    sns.set(style='white', font_scale=1.5)

    grid = sns.JointGrid(
        x=parameter_1,
        y=parameter_2,
        space=0,
        height=height
    )
    grid.plot_joint(
        sns.kdeplot,
        clip=(
            (parameter_1.min(), parameter_1.max()),
            (parameter_2.min(), parameter_2.max())),
        fill=True,
        thresh=0,
        levels=100,
        cmap='rocket'
    )
    grid.plot_marginals(
        sns.histplot,
        color='#03051A',
        bins=25
    )
    grid.set_axis_labels(parameter_1_name, parameter_2_name)

    plt.show()
    return None


def visualize_heatmap(df, pivot_varaibles, rounding=4, figsize=(10, 10),
                      **kwargs):
    """
    """
    columns = [i for i in pivot_varaibles]
    pivoted = df[columns].round(rounding)
    pivoted = pivoted.pivot(
        pivot_varaibles[0],
        pivot_varaibles[1],
        pivot_varaibles[2]
    )

    sns.set(style='white', font_scale=1.5)
    plt.figure(figsize=figsize)

    sns.heatmap(
        pivoted,
        **kwargs
    )

    plt.show()
    return None
