import numpy as np
from scipy.stats import binom, halfnorm

import pandas as pd

from modules.stats.backend import compute_sigmoid


def generate_poisson_ar(lam_int, slope_a, slope_b, mu_noise, sigma_noise,
                        burn_factor=2, time_steps=48):
    """
    """
    init = np.random.poisson(lam_int)
    slope = np.random.beta(slope_a, slope_b)
    process = [init]
    true_parameters = {
        'mu_noise': mu_noise,
        'sigma_noise': sigma_noise,
        'Slope': slope
    }
    for time_step in range(time_steps * burn_factor):

        new_lam = (slope * process[time_step]) + \
            np.random.normal(mu_noise, sigma_noise)
        new_lam = max(0, new_lam)
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
    for player in range(players):

        player_ability = np.random.normal(
             np.random.normal(0, 2, 1)[0],
            halfnorm.rvs(loc=0, scale=2, size=1)[0],
            1
        )[0]

        for level in range(levels):

            level_difficulty = np.random.normal(
                np.random.normal(0.5, 2, 1)[0],
                halfnorm.rvs(loc=0, scale=2, size=1)[0],
                1
            )[0]

            p_success = compute_sigmoid(player_ability - level_difficulty)

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

    df = pd.DataFrame(
        data
    )
    return df
