from abc import ABC, abstractmethod

import numpy as np

import pymc3 as pm

import matplotlib.pyplot as plt

from modules.visuals import visualize_regression_lines

class AbastractModel(ABC):
    """
    """
    @abstractmethod
    def generate_model(self):
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

        self.generate_model(
            X=self.X,
            y=self.y
        )

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

    def show_prior_summary(self, n_lines=100, figsize=(10, 8), **kwargs):
        """
        """
        visualize_regression_lines(
            X=self.X,
            y=self.y,
            intercepts=self.prior_checks['Intercpet'],
            slopes=self.prior_checks['Slope'],
            overlay=True,
            n_lines=n_lines,
            figsize=figsize,
            title='Prior Regression Lines',
            **kwargs
        )

    def show_posterior_summary(self, figsize=(10, 8), **kwargs):
        """
        """
        if self.map:
            self.print_model_summary()
            visualize_regression_lines(
                X=self.X,
                y=self.y,
                intercepts=[self.map_estimate['Intercpet']],
                slopes=[self.map_estimate['Slope']],
                figsize=figsize,
                overlay=False,
                title='Posterior MAP Regression Line',
                n_lines=1,
                **kwargs
            )
        else:
            self.print_model_summary()
            with self.model:
                pm.plot_trace(self.traces)

    def show_plate(self):
        """
        """
        plate = pm.model_graph.model_to_graphviz(
            self.model.model
        )
        return plate
