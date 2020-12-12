import numpy as np

import pymc3 as pm

from modules.utils.models_utils import AbastractModel


class CompoundModel(AbastractModel):
    """
    """
    def __init__(self, X, y):
        """
        """
        self.X = X
        self.y = y
        self.model = self.generate_model(X, y)

    def generate_model(self, X, y):
        """
        """
        with pm.Model() as compound_model:

            precursor = pm.Normal(
                name='Precursor',
                mu=10,
                sd=2
            )

            multiplier_x = pm.Normal(
                name='Multiplier x',
                mu=5,
                sigma=1
            )
            multiplier_y = pm.Normal(
                name='Multiplier y',
                mu=2,
                sigma=1
            )

            chemical_x = pm.Normal(
                name='Chemical x',
                mu=precursor * multiplier_x,
                sigma=1,
            )
            chemical_y = pm.Normal(
                name='Chemical y',
                mu=precursor * multiplier_y,
                sigma=1,
            )

            days_slope = pm.Beta(
                name='Days Slope',
                alpha=2,
                beta=2
            )

            chemical_z = chemical_y + chemical_x - (days_slope * X)

            saveoura900 = pm.Normal(
                name='SaveourA900',
                mu=chemical_z,
                sigma=pm.Exponential(lam=1, name='Sigma'),
                observed=y
            )

        return compound_model

    def show_prior_summary(self, **kwargs):
        """
        """
        raise NotImplementedError('Method not yet implemented')

    def show_posterior_summary(self, parameters_name, figsize=(10, 8),
                               **kwargs):
        """
        """
        self.print_model_summary(parameters_name=parameters_name)
        if not self.map:
            with self.model:
                pm.plot_trace(
                    self.traces,
                    var_names=parameters_name,
                    **kwargs
                )


class GameDifficultyModel(AbastractModel):
    """
    """
    def __init__(self, players_id, levels_id, num_attempts, num_success):
        """
        """
        self.model = self.generate_model(
            players_id=players_id,
            levels_id=levels_id,
            num_attempts=num_attempts,
            num_success=num_success
        )

    def generate_model(self, players_id, levels_id, num_attempts, num_success):
        """
        """
        coords = {
            'Players': np.unique(players_id),
            'Levels': np.unique(levels_id),
            'obs_id': np.arange(num_success.size)
        }
        with pm.Model(coords=coords) as model:
            players_idx = pm.Data(
                'players_idx',
                players_id,
                dims='obs_id'
            )
            levels_idx = pm.Data(
                'levels_idx',
                levels_id,
                dims='obs_id'
            )

            # hyper priors
            hyper_player_mu = pm.Normal(
                name='player_ability_mu',
                mu=0,
                sigma=0.5
            )
            hyper_level_mu = pm.Normal(
                name='level_difficulty_mu',
                mu=0,
                sigma=0.5
            )
            hyper_sigma = pm.HalfNormal(
                name='hyper_sigma',
                sigma=1.
            )

            # latent variables
            player_ability = pm.Normal(
                name='player_ability',
                mu=hyper_player_mu,
                sigma=hyper_sigma,
                dims='Players'
            )
            level_difficulty = pm.Normal(
                name='level_difficulty',
                mu=hyper_level_mu,
                sigma=hyper_sigma,
                dims='Levels'
            )

            fixed_intercept = pm.Normal(
                name='fixed_intercept',
                mu=0,
                sigma=1,
                dims='Levels'
            )

            # probability of success increase when ability is greater than
            # level difficulty
            delta = pm.Deterministic(
                'delta = player_ability - level_difficulty',
                fixed_intercept +
                player_ability[players_idx] + level_difficulty[levels_idx]
            )

            probability_success = pm.Deterministic(
                'p = invlogit(delta)',
                pm.math.invlogit(delta)
            )

            outcome = pm.Binomial(
                'observed_success',
                p=probability_success,
                n=num_attempts,
                observed=num_success,
                dims='obs_id'
            )

        return model

    def show_prior_summary(self, **kwargs):
        """
        """
        raise NotImplementedError('Method not yet implemented')

    def show_posterior_summary(self, parameters_name, figsize=(10, 8),
                               **kwargs):
        """
        """
        self.print_model_summary(
            parameters_name=[
                'player_ability_mu',
                'hyper_sigma',
                'level_difficulty_mu'
            ]
        )
        if not self.map:
            with self.model:
                pm.plot_trace(
                    data=self.traces,
                    var_names=parameters_name,
                    **kwargs
                )

        return None

    def show_forest_plot(self, parameters_name, **kwargs):
        """
        """
        with self.model:
            pm.plot_forest(
                data=self.traces,
                var_names=parameters_name,
                combined=True
            )

        return None
