import numpy as np

import pymc3 as pm

from modules.utils.models_utils import AbastractModel


class BivariateNormalRegression(AbastractModel):
    """
    """
    def __init__(self, X, y, intercept_prior=(0, 100), slope_prior=(0, 100),
                 likelihood_sigma_prior=100, fit_intercept=True):
        """
        """
        self.intercept_prior = intercept_prior
        self.slope_prior = slope_prior
        self.likelihood_sigma_prior = likelihood_sigma_prior
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
                name='likelihood Sigma',
                lam=self.likelihood_sigma_prior
            )

            likelihood = pm.Normal(
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

            likelihood = pm.Bernoulli(
                name='y',
                p=p,
                observed=y
            )

        return model

class PoissonAR1(AbastractModel):
    """
    """
    def __init__(self, X, y, slope_prior, innovation_prior):
        """
        """
        self.slope_prior = slope_prior
        self.innovation_prior = innovation_prior
        self.X = X
        self.y = y
        self.model = self.generate_model(X, y)

    def generate_model(self, X, y):
        """
        """
        with pm.Model() as ar_model:

            slope = pm.Beta(
                alpha=self.slope_prior[0],
                beta=self.slope_prior[1],
                name='Slope'
            )

            innovation = pm.Normal(
                mu=self.innovation_prior[0],
                sigma=self.innovation_prior[1],
                name='Innovation'
            )

            lam = slope*X + innovation
            lam = pm.math.maximum(0, lam)

            outcome = pm.Poisson(
                mu=lam,
                observed=y,
                name='y'
            )

        return ar_model

    def show_posterior_summary(self, figsize=(10, 8), **kwargs):
        """
        """
        self.print_model_summary()
        if not self.map:
            with self.model:
                pm.plot_trace(self.traces)

class PolynomialRegression(AbastractModel):
    """
    """
    def __init__(self, X, y, cubic=False, intercept_prior=(0, 1),
                 slopes_prior=(0, 1), sigma_prior=1):
        """
        """
        self.X = X
        self.y = y
        self.cubic = cubic
        self.intercept_prior = intercept_prior
        self.slopes_prior = slopes_prior
        self.sigma_prior = sigma_prior
        self.model = self.generate_model(X, y, cubic)

    def generate_model(self, X, y, cubic=False):
        """
        """
        with pm.Model() as polynomial_model:

            intercept = pm.Normal(
                name='Intercept',
                mu=self.intercept_prior[0],
                sd=self.intercept_prior[1]
            )
            slope = pm.Normal(
                name='Slope',
                mu=self.slopes_prior[0],
                sd=self.slopes_prior[1]
            )
            slope_1 = pm.Normal(
                name='Slope_1',
                mu=self.slopes_prior[0],
                sd=self.slopes_prior[1]
            )

            if not cubic:
                mu = intercept + X*slope + np.power(X, 2)*slope_1
            else:
                slope_2 = pm.Normal(
                    name='Slope_2',
                    mu=self.slopes_prior[0],
                    sd=self.slopes_prior[1]
                )
                mu = intercept + X*slope + np.power(X, 2)*slope_1 + \
                    np.power(X, 3)*slope_2

            sigma = pm.Exponential(
                name='Sigma',
                lam=self.sigma_prior
            )

            likelihood = pm.Normal(
                name='y',
                mu=mu,
                sd=sigma,
                observed=y
            )

            return polynomial_model


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
