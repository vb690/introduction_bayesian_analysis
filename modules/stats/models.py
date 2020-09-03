from scipy.stats import binom
import pymc3 as pm

from modules.stats.backend import compute_sp_posterior


class PrevalenceModel:
    """
    """
    def __init__(self, parameter_space):
        """
        """
        self.parameter_space = parameter_space

    def fit(self, k, n, prior):
        """
        """
        likelyhood = binom(k, n).pmf(self.parameter_space)
        posterior = compute_sp_posterior(
            prior=prior,
            likelyhood=likelyhood
        )
        setattr(self, 'prior', prior)
        setattr(self, 'likelyhood', likelyhood)
        setattr(self, 'posterior')

    def sample(self, n_samples=1000):
        """
        """
        samples = np.random.choice(
            self.parameter_space,
            n_samples,
            p=self.posterior
        )
        return samples


class BivariateLogitNormalRegression:
    """
    """
    def __init__(self, slope_prior, intercept_prior=None):
        """
        """
        self.intercept_prior = intercept_prior
        self.slope_prior = slope_prior

    def __generate_model(self, X, y):
        """
        """
        with pm.Model() as logistic_model:
            if self.intercept_prior is None:
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
                name='std_normal',
                sigma=2
            )

            pm.Normal(
                name='logit_normal',
                mu=mu,
                sd=sigma,
                observed=y
           )

        setattr(self, 'model', logistic_model)
        return None

    def fit(self, X, y, MAP=True, **kwargs):
        """
        """
        self.__generate_model(
            X=X,
            y=y
        )
        with self.model:
            if MAP:
                map_estimate = pm.find_MAP()
                return map_estimate
            else:
                traces = pm.sample(**kwargs)
                return traces

    def show_plate(self):
        """
        """
        plate = pm.model_graph.model_to_graphviz(
            self.model.model
        )
        return plate
