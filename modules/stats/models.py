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

    def generate_model(self, X, y):
        """
        """
        with pm.Model() as model:
            if not self.fit_intercept:
                intercept = pm.math.constant(
                    0,
                    name='Intercpet'
                )
            else:
                intercept = pm.Normal(
                    name='Intercpet',
                    mu=self.intercept_prior[0],
                    sd=self.intercept_prior[1]
                )
            slope = pm.Normal(
                name='Slope',
                mu=self.slope_prior[0],
                sd=self.slope_prior[1]
            )

            mu = intercept + slope * X

            sigma = pm.HalfNormal(
                name='Likelyhood Sigma',
                sigma=self.likelyhood_sigma_prior
            )

            likelyhood = pm.Normal(
                name='y',
                mu=mu,
                sd=sigma,
                observed=y
           )

            prior_checks = pm.sample_prior_predictive(
                samples=100
            )
            setattr(self, 'prior_checks', prior_checks)

        return model
