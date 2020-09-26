import numpy as np

import pymc3 as pm

from modules.utils.models_utils import AbastractModel


class BivariateRegression(AbastractModel):
    """
    """
    def __init__(self, X, y, intercept_prior=(0, 100), slope_prior=(0, 100),
                 likelihood_sigma_prior=100, fit_intercept=True,
                 logistic=False):
        """
        """
        self.intercept_prior = intercept_prior
        self.slope_prior = slope_prior
        self.likelihood_sigma_prior = likelihood_sigma_prior
        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y
        self.logistic = logistic
        self.model = self.generate_model(X, y)

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

            if self.logistic:
                p = pm.Deterministic(
                    'p ~ Sigmoid(Intercept + Slope*X)',
                    pm.math.sigmoid(intercept + slope * X)
                )
                likelihood = pm.Bernoulli(
                    name='y',
                    p=p,
                    observed=y
                )

            else:
                mu = pm.Deterministic(
                    'mu ~ Intercept + Slope*X',
                    intercept + slope * X
                )

                sigma = pm.HalfCauchy(
                    name='Sigma',
                    beta=self.likelihood_sigma_prior
                )

                likelihood = pm.Normal(
                    name='y',
                    mu=mu,
                    sd=sigma,
                    observed=y
                )

        return model


class PoissonAR1(AbastractModel):
    """
    """
    def __init__(self, X, y, intercept_prior, slope_prior):
        """
        """
        self.intercept_prior = intercept_prior
        self.slope_prior = slope_prior
        self.X = X
        self.y = y
        self.model = self.generate_model(X, y)

    def generate_model(self, X, y):
        """
        """
        with pm.Model() as ar_model:

            intercept = pm.Normal(
                mu=self.intercept_prior,
                name='Intercept'
            )
            slope = pm.Beta(
                alpha=self.slope_prior[0],
                beta=self.slope_prior[1],
                name='Slope'
            )

            lam = pm.Deterministic(
                'lambda ~ Intercept + Slope*yt-1',
                intercept + slope*X
            )

            outcome = pm.Poisson(
                mu=lam,
                observed=y,
                name='y'
            )

        return ar_model

    def show_posterior_summary(self, parameters_name, figsize=(10, 8),
                               **kwargs):
        """
        """
        self.print_model_summary(parameters_name=parameters_name)
        if not self.map:
            with self.model:
                pm.plot_trace(self.traces, compact=True)


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
            slope1 = pm.Normal(
                name='Slope_1',
                mu=self.slopes_prior[0],
                sd=self.slopes_prior[1]
            )

            if not cubic:
                mu = pm.Deterministic(
                    'mu ~ Intercept + Sl*X + Sl_1*(X**2)',
                    intercept + X*slope + np.power(X, 2)*slope1
                )
            else:
                slope2 = pm.Normal(
                    name='Slope_2',
                    mu=self.slopes_prior[0],
                    sd=self.slopes_prior[1]
                )
                mu = pm.Deterministic(
                    'mu ~ Intercept + Sl*X + Sl_1*(X**2) + Sl_2*(X**3)',
                    intercept +
                    X*slope + np.power(X, 2)*slope1 + np.power(X, 3)*slope2
                )
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
