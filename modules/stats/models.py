from scipy.stats import binom
import pymc3 as pm

from modules.stats.backend import compute_sp_posterior
from modules.utils.models_utils import AbastractModel


class BivariateNormalRegression(AbastractModel):
    """
    """
    def __init__(self, X, y, intercept_prior=(0, 100), slope_prior=(0, 100),
                 likelyhood_sigma_prior=100, fit_intercept=True):
        """
        """
        self.intercept_prior = intercept_prior
        self.slope_prior = slope_prior
        self.likelyhood_sigma_prior = likelyhood_sigma_prior
        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y
        self.model = self.generate_model(X, y)
        self.logistic = False

    def generate_model(self, X, y):
        """
        """
        with pm.Model() as model:
            if not self.fit_intercept:
                intercept = pm.math.constant(
                    0,
                    name='Intercept'
                )
            else:
                intercept = pm.Normal(
                    name='Intercept',
                    mu=self.intercept_prior[0],
                    sd=self.intercept_prior[1]
                )
            slope = pm.Normal(
                name='Slope',
                mu=self.slope_prior[0],
                sd=self.slope_prior[1]
            )

            mu = intercept + slope * X

            sigma = pm.Exponential(
                name='Likelyhood Sigma',
                lam=self.likelyhood_sigma_prior
            )

            likelyhood = pm.Normal(
                name='y',
                mu=mu,
                sd=sigma,
                observed=y
           )

        return model


class BivariateLogisticRegression(AbastractModel):
    """
    """
    def __init__(self, X, y, intercept_prior=(0, 100), slope_prior=(0, 100),
                 fit_intercept=True):
        """
        """
        self.intercept_prior = intercept_prior
        self.slope_prior = slope_prior
        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y
        self.model = self.generate_model(X, y)
        self.logistic = True

    def generate_model(self, X, y):
        """
        """
        with pm.Model() as model:
            if not self.fit_intercept:
                intercept = pm.math.constant(
                    0,
                    name='Intercept'
                )
            else:
                intercept = pm.Normal(
                    name='Intercept',
                    mu=self.intercept_prior[0],
                    sd=self.intercept_prior[1]
                )
            slope = pm.Normal(
                name='Slope',
                mu=self.slope_prior[0],
                sd=self.slope_prior[1]
            )

            theta = intercept + slope * X

            p = pm.math.sigmoid(theta)

            likelyhood = pm.Bernoulli(
                name='y',
                p=p,
                observed=y
           )

        return model


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

    def show_prior_summary(self, X, y):
        """
        """
        raise NotImplementedError('Method not yet implemented')

    def show_posterior_summary(self, figsize=(10, 8), **kwargs):
        """
        """
        self.print_model_summary()
        if not self.map:
            with self.model:
                pm.plot_trace(self.traces)
