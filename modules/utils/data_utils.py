import numpy as np
from scipy.stats import binom, halfnorm

import pandas as pd

from modules.stats.backend import compute_sigmoid


def generate_polynomial_data(X,  degree=3, noise_ratio=1, batches=1):
    """
    """
    intercept = np.random.normal()
    coefficients = {
        f'slope_{coef}': np.random.normal() for coef in range(1, degree+1)
    }
    true_parameters = {**coefficients, **{'intercept': intercept}}

    df = []
    for batch in range(batches):

        y = []
        df_batch = pd.DataFrame(columns=['y', 'X'])

        for x in X:

            mu = intercept

            for coef in range(1, degree+1):

                mu += coefficients[f'slope_{coef}']*(x**coef)

            y.append(float(mu))

        y = np.array(y)
        y += np.random.normal(y.mean(), y.std() * noise_ratio, len(y))
        y = (y - y.mean()) / y.std()

        df_batch['y'] = y
        df_batch['X'] = X
        df.append(df_batch)

    df = pd.concat(df)
    return df, true_parameters


def generate_poisson_ar_data(lam_int, slope_a, slope_b, burn_factor=2,
                             time_steps=48, batches=1):
    """
    """
    intercept = np.random.poisson(lam_int)
    slope = np.random.beta(slope_a, slope_b)
    process = [intercept]
    true_parameters = {
        'intercept': intercept,
        'slope': slope
    }
    for time_step in range(time_steps * burn_factor):

        new_lam = intercept + (slope * process[time_step])
        new_value = np.random.poisson(new_lam)
        process.append(new_value)

    process = np.array(process[-time_steps:])
    return process, true_parameters


def generate_game_difficulty_data(players=100, levels=10, n_sessions=10,
                                  max_attempts=10):
    """
    """
    data = {
        'player_id': [],
        'level': [],
        'session': [],
        'num_success': [],
        'num_attempts': [],
    }
    true_parameters = {
        'player_id': [],
        'levels_id': [],
        'level_difficulty': [],
        'player_ability': [],
        'delta': [],
        'probability_success':[]

    }
    players_ability = []
    for player in range(players):

        player_ability = np.random.normal(
             np.random.normal(0, 0.5, 1)[0],
            halfnorm.rvs(loc=0, scale=1, size=1)[0],
            1
        )[0]
        players_ability.append(player_ability)


    levels_difficulty = []
    for level in range(levels):

        level_difficulty = np.random.normal(
            np.random.normal(0, 0.5, 1)[0],
            halfnorm.rvs(loc=0, scale=1, size=1)[0],
            1
        )[0]
        levels_difficulty.append(level_difficulty)

    for player, ability in enumerate(players_ability):

        for level, difficulty in enumerate(levels_difficulty):

            delta = ability - difficulty
            p_success = compute_sigmoid(
                delta
            )

            true_parameters['player_id'].append(player)
            true_parameters['levels_id'].append(level)
            true_parameters['level_difficulty'].append(difficulty)
            true_parameters['player_ability'].append(ability)
            true_parameters['delta'].append(delta)
            true_parameters['probability_success'].append(p_success)

            for session in range(n_sessions):

                attempts = np.random.randint(1, max_attempts, 1)[0]
                data['player_id'].append(player)
                data['level'].append(level)
                data['session'].append(session)
                data['num_success'].append(
                    binom.rvs(
                        n=attempts,
                        p=p_success
                    )
                )
                data['num_attempts'].append(attempts)

    true_parameters = pd.DataFrame(
        true_parameters
    )
    df = pd.DataFrame(
        data
    )
    return df, true_parameters
