# Bayesian data analysis: an introductory workshop
  
<p align="center">
  <img width="906" height="300" src="https://github.com/vb690/introduction_bayesian_analysis/blob/master/results/figures/presentation/header.png">
</p>

Workshop on introductory concepts in bayesian data analysis

# Notebooks 
The workshop comes in two slightly different versions: 

## University Workshop
Version presented at the IGGI Conference 2020
```
* Introduction  

* Bayesian Gears  
    - From counts to probability  
    - Bayesian updating  
    - Likelihood, Parameters, Prior and Posterior  
    
* Bayesian Machinery  
    - Parameters Estimation  
    - Grid Search, Quadratic Approximation, MCMC  

* Bayesian Models  
    - PyMC3 Model Building  
    - Linear Regression  
    - Logistic Regression  
    - Graphical Models  
```

## Applied Workshop
Application-oriented version, more suitable for being delivered in industry settings
```
* Introduction

* Bayesian Approach to Inference
  - Counts
  - Updating Counts
  - From Counts to Probability
  - Likelihood, Parameters, Prior and Posterior
  - Parameters Estimation
  - Bayesian Models

* PyMC3
  - Model Building
  - Model Inspecting
  - Model Fitting
  - Model Evaluating and Comparing
  - Model Predicting

* Applications
  - PyMC3 vs scikit-learn
  - Web Traffic Estimation
  - Advertising Effect on Revenue
  - Game Difficulty Estimation
  - Model Comparison
```

# Links 

* https://xcelab.net/rm/statistical-rethinking/ (most of the content has been adapted from here)
* https://docs.pymc.io/
* https://en.wikipedia.org/wiki/Papers,_Please (images and narrative framework for the University Workshop version)

# Requirements 
* Download your local version of the workshop repository
* Install anaconda https://docs.anaconda.com/anaconda/install/
* Open the Anaconda Powershell Prompt in the workshop directory
* Then from the Prompt
``` python
# create anaconda environment
conda create -n workshop_env python=3.6

# activate the environment
conda activate workshop_env

# install the requirements
conda install --file requirements.txt

# open jupyter 
jupyter notebook
```
