from abc import ABC, abstractmethod

import numpy as np

import pymc3 as pm

import matplotlib.pyplot as plt

from modules.visuals import visualize_regression_lines


class AbastractModel(ABC):
    """
    """
    @abstractmethod
    def generate_model(self, X, y):
        """
        """
        raise NotImplementedError('Subclass Model has to implement \
                                   generate_model method')

    def print_model_summary(self):
        """
        """
        if self.map:
            print('Maximum A Posteriori')
            print('')
            for parameter_name, parameter_map in self.map_estimate.items():

                print(f'{parameter_name}: {parameter_map}')
        else:
            print('MCMC Estimates')
            print('')
            with self.model:
                summary = pm.summary(self.traces)
            summary = summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']]
            print(summary)

    def riparametrize_priors(self, new_parameters):
        """
        """
        for parameter_name, parameter_values in new_parameters.items():

            setattr(self, parameter_name, parameter_values)

        model = self.generate_model(
            X=self.X,
            y=self.y
        )
        setattr(self, 'model', model)

    def fit(self, X, y, MAP=True, **kwargs):
        """
        """
        setattr(self, 'map', MAP)
        with self.model:
            if MAP:
                map_estimate = pm.find_MAP()
                setattr(self, 'map_estimate', map_estimate)
            else:
                traces = pm.sample(**kwargs)
                setattr(self, 'traces', traces)

    def predict(self, X, y):
        """
        """
        with self.generate_model(X, y):
            if self.map:
                posterior_predictions = pm.sample_posterior_predictive(
                    self.map_estimate
                )
            else:
                posterior_predictions = pm.sample_posterior_predictive(
                    self.traces
                )
        setattr(self, 'posterior_predictions', posterior_predictions)
        return posterior_predictions

    def show_prior_summary(self, n_lines=100, figsize=(10, 8), **kwargs):
        """
        """
        with self.model:
            prior_checks = pm.sample_prior_predictive(
                samples=n_lines
            )
            setattr(self, 'prior_checks', prior_checks)

        visualize_regression_lines(
            X=self.X,
            y=self.y,
            intercepts=self.prior_checks['Intercept'],
            slopes=self.prior_checks['Slope'],
            overlay=True,
            figsize=figsize,
            title='Prior Regression Lines',
            logistic=self.logistic,
            **kwargs
        )

    def show_posterior_summary(self, figsize=(10, 8), **kwargs):
        """
        """
        self.print_model_summary()
        if self.map:
            visualize_regression_lines(
                X=self.X,
                y=self.y,
                intercepts=[self.map_estimate['Intercept']],
                slopes=[self.map_estimate['Slope']],
                figsize=figsize,
                overlay=False,
                predictions=None,
                title='Posterior MAP Regression Line',
                logistic=self.logistic,
                **kwargs
            )
        else:
            with self.model:
                posterior_checks = pm.sample_posterior_predictive(
                    self.traces,
                    var_names=['Intercept', 'Slope', 'y']
                )
                setattr(self, 'posterior_checks', posterior_checks)
                pm.plot_trace(self.traces)
            visualize_regression_lines(
                X=self.X,
                y=self.y,
                intercepts=self.posterior_checks['Intercept'],
                slopes=self.posterior_checks['Slope'],
                figsize=figsize,
                overlay=False,
                predictions=posterior_checks['y'],
                title='Posterior Regression Lines',
                logistic=self.logistic,
                **kwargs
            )

    def show_plate(self):
        """
        """
        plate = pm.model_graph.model_to_graphviz(
            self.model.model
        )
        return plate
