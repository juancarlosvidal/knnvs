"""
Simulation study for comparing CDF estimation methods.

This module performs a comprehensive comparison of five different methods for estimating
cumulative distribution functions (CDFs) in heteroscedastic regression settings:

1. Empirical CDF based on K-NN with feature selection (ECDF1)
2. Empirical CDF based on K-NN without feature selection (ECDF2)
3. Normal CDF with mean and variance estimated by K-NN with feature selection (NCDF1)
4. Normal CDF with mean and variance estimated by K-NN without feature selection (NCDF2)
5. GAMLSS (Generalized Additive Models for Location, Scale and Shape) from R

The simulation generates synthetic data under various scenarios with different mean and
variance structures, then evaluates how well each method estimates the true CDF.

Usage:
    python sim_cdf.py -s 12 -a 0.05

    Arguments:
        -s, --scenario: Simulation scenario number (1-12), default=12
        -a, --alpha: Significance level for variable selection, default=0.05

Output:
    - CSV file: cdf_scenario_{scenario}.csv with MSE metrics for each method
    - PDF plots: scenario_{scenario}_{n}_{p}_{sim}_{obs}.pdf comparing all CDFs

Author: Juan
Date: 2025
"""

import math
import argparse

import logging
import csv

import random
import numpy as np
import pandas as pd
from scipy.stats import norm
import sklearn.metrics as skl

from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects.conversion import localconverter
import rpy2.rinterface_lib.callbacks

from knnvs import initialize_knn, no_initialize_knn

import matplotlib.pyplot as plt


logging.basicConfig(format='%(message)s', level=logging.INFO)
# logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
# logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Store the original R output and message handlers
original_writeconsole = rpy2.rinterface_lib.callbacks.consolewrite_print
original_writeconsole_warnerror = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
original_writeconsole_flush = rpy2.rinterface_lib.callbacks.consoleflush


def suppress_r_output(text):
    """
    Suppress R console output.

    Custom callback function that discards all R console output to keep
    Python logs clean during simulation runs.

    Args:
        text (str): Text output from R console (ignored).
    """
    pass

# Redirect the R console output and messages to the custom suppress function
rpy2.rinterface_lib.callbacks.consolewrite_print = suppress_r_output
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = suppress_r_output
rpy2.rinterface_lib.callbacks.consoleflush = suppress_r_output
rpy2.rinterface_lib.callbacks.consolewrite_message = suppress_r_output


def sim_scenario(z, scenario):
    """
    Define mean and variance functions for different simulation scenarios.

    This function implements 12 different data-generating processes with varying
    mean and variance structures. Each scenario models heteroscedastic regression
    with Y ~ N(f0(X), g0(X)), where f0 is the mean function and g0 is the variance function.

    Args:
        z (np.ndarray): Covariate vector of shape (p,).
        scenario (int): Scenario number (1-12).

    Returns:
        tuple: (f0, g0) where:
            - f0 (float): Mean value E[Y|X=z]
            - g0 (float): Variance value Var[Y|X=z]

    Scenarios:
        1: Mean depends on z[1:3], constant variance (homoscedastic)
        2: Constant mean, variance depends on z[0] (pure heteroscedastic)
        3: Mean depends on z[1:3], variance depends on z[0]
        4: Mean depends on z[0:4], constant variance
        5: Constant mean, variance depends on z[0:4]
        6: Mean depends on z[0:3], variance depends on z[3:5]
        7: Mean depends on z[0:4], variance depends on z[1:5]
        8: Mean depends on z[0:8], constant variance (high-dimensional)
        9: Constant mean, variance depends on z[0:8] (high-dimensional)
        10: Mean depends on z[0:6], variance depends on z[7:10]
        11: Mean depends on z[0:4], variance depends on z[5:10]
        12: Exponential mean function, linear variance (non-linear)

    Examples:
        >>> z = np.array([0.5, 0.3, 0.7, 0.2])
        >>> f0, g0 = sim_scenario(z, scenario=1)
        >>> print(f"Mean: {f0}, Variance: {g0}")
        Mean: 5.0, Variance: 1
    """
    if scenario == 1:
        f0 = 5 * np.sum(z[1:3])
        g0 = 1
    elif scenario == 2:
        f0 = 0
        g0 = 5 * z[0]
    elif scenario == 3:
        f0 = 5 * np.sum(z[1:3])
        g0 = 5 * z[0]
    elif scenario == 4:
        f0 = 5 * np.sum(z[0:4])
        g0 = 1
    elif scenario == 5:
        f0 = 0
        g0 = 5 * np.sum(z[0:4])
    elif scenario == 6:
        f0 = 5 * np.sum(z[0:3])
        g0 = 5 * np.sum(z[3:5])
    elif scenario == 7:
        f0 = 5 * np.sum(z[0:4])
        g0 = 5 * np.sum(z[1:5])
    elif scenario == 8:
        f0 = 5 * np.sum(z[0:8])
        g0 = 1
    elif scenario == 9:
        f0 = 0
        g0 = 5 * np.sum(z[0:8])
    elif scenario == 10:
        f0 = 5 * np.sum(z[0:6])
        g0 = 5 * np.sum(z[7:10])
    elif scenario == 11:
        f0 = 5 * np.sum(z[0:4])
        g0 = 5 * np.sum(z[5:10])
    elif scenario == 12:
        f0 = np.exp(3 * np.sum(z[0:2]))
        g0 = 1 * z[3] + 1 * z[4]
    else:
        f0 = 0
        g0 = 0
    return f0, g0


def sim_config(scenario):
    """
    Configure sample sizes and dimensions for a given scenario.

    Returns appropriate combinations of sample sizes (n) and number of covariates (p)
    to simulate for each scenario. Different scenarios use different dimensionality
    settings based on their complexity.

    Args:
        scenario (int): Scenario number (1-12).

    Returns:
        tuple: (n0, p0) where:
            - n0 (list): Sample sizes to simulate [5000, 10000, 20000, 50000, 100000]
            - p0 (list): Number of covariates to simulate
                - [5, 10, 20, 25] for scenarios 4-7 (moderate dimensions)
                - [10, 25, 50, 100] for other scenarios (higher dimensions)

    Examples:
        >>> n_list, p_list = sim_config(scenario=1)
        >>> print(f"Sample sizes: {n_list}")
        Sample sizes: [5000, 10000, 20000, 50000, 100000]
        >>> print(f"Dimensions: {p_list}")
        Dimensions: [10, 25, 50, 100]
    """
    n0, p0 = [5000, 10000, 20000, 50000, 100000], [3, 10, 20, 25]
    if 4 <= scenario < 8:
        p0 = [5, 10, 20, 25]
    else:
        p0 = [10, 25, 50, 100]
    return n0, p0


def simulate(n, d, scenario):
    """
    Generate synthetic data for a given scenario.

    Simulates data from a heteroscedastic regression model:
        Y_i = f0(X_i) + sqrt(g0(X_i)) * epsilon_i

    where X_i ~ Uniform(0,1)^d and epsilon_i ~ N(0,1) are independent.
    The functions f0 and g0 are defined by the scenario.

    Args:
        n (int): Sample size (number of observations).
        d (int): Number of covariates/features.
        scenario (int): Scenario number (1-12) determining f0 and g0.

    Returns:
        tuple: (x, y) where:
            - x (np.ndarray): Covariate matrix of shape (n, d), dtype float32
            - y (np.ndarray): Response vector of shape (n,), dtype float32

    Examples:
        >>> np.random.seed(42)
        >>> x, y = simulate(n=100, d=5, scenario=1)
        >>> print(f"X shape: {x.shape}, Y shape: {y.shape}")
        X shape: (100, 5), Y shape: (100,)
    """
    x = np.random.uniform(0, 1, (n, d)).astype('float32')
    eps = np.random.standard_normal(n)
    y = np.zeros(n).astype('float32')
    for j in range(n):
        f0, g0 = sim_scenario(x[j], scenario)
        y[j] = f0 + math.sqrt(g0) * eps[j]
    return x, y


def assess(ground_truth, selected_vars):
    """
    Evaluate variable selection performance.

    Computes accuracy and false positive rate for variable selection methods.
    Note: This function is currently not used in the main simulation loop.

    Args:
        ground_truth (set): Set of indices of true relevant variables.
        selected_vars (set): Set of indices of variables selected by the method.

    Returns:
        tuple: (accuracy, fpr) where:
            - accuracy (float): Proportion of true variables that were selected
                               = |selected ∩ truth| / |truth|
            - fpr (float): Proportion of selected variables that are false positives
                          = |selected - truth| / |selected|

    Examples:
        >>> ground_truth = {0, 1, 2}
        >>> selected_vars = {0, 1, 3, 4}
        >>> acc, fpr = assess(ground_truth, selected_vars)
        >>> print(f"Accuracy: {acc:.2f}, FPR: {fpr:.2f}")
        Accuracy: 0.67, FPR: 0.50
    """
    # Calculate Accuracy
    accuracy = len(selected_vars.intersection(ground_truth)) / len(ground_truth)
    # False Positive Rate (FPR): FP / (FP + TN)
    fpr = len(selected_vars - ground_truth) / len(selected_vars)
    return accuracy, fpr


if __name__ == '__main__':
    """
    Main execution block for CDF estimation comparison study.

    This simulation study:
    1. Generates synthetic data according to the specified scenario
    2. Trains K-NN models with and without feature selection
    3. Trains GAMLSS model using R
    4. For each observation, estimates the CDF using 5 different methods
    5. Compares each estimated CDF against the true theoretical CDF
    6. Saves results (MSE metrics) to CSV and visualization plots to PDF

    The simulation loops over:
    - Different sample sizes (n)
    - Different dimensionalities (p)
    - Monte Carlo repetitions
    - Individual observations in each dataset
    """
    parser = argparse.ArgumentParser(
        description='CDF Estimation Comparison Study for Heteroscedastic Regression')
    parser.add_argument('-s', '--scenario', default="12", type=int,
                       help='Simulation scenario number (1-12)')
    parser.add_argument('-a', '--alpha', default="0.05", type=float,
                       help='Significance level for variable selection (default: 0.05)')
    args = parser.parse_args()

    _scenario = args.scenario
    _alpha = args.alpha

    # Set random seeds for reproducibility
    random.seed(1)
    np.random.seed(1)

    # Enable the automatic conversion between NumPy and R objects
    numpy2ri.activate()
    # Enable the automatic conversion between Pandas and R objects
    pandas2ri.activate()

    # Load R script for GAMLSS models
    with open('/Users/juan/PycharmProjects/knn-big/knnvar/script.r', 'r') as f:
        string = f.read()
    # Parse R code using SignatureTranslatedAnonymousPackage (STAP)
    r_package = SignatureTranslatedAnonymousPackage(string, "my_package")

    # Simulation configuration
    _n_sims = 1  # Number of Monte Carlo repetitions (can be increased)
    _n_list, _p_list = sim_config(_scenario)  # Get n and p values to simulate
    _k = 100  # Fixed number of neighbors for CDF prediction

    _str_sim = 'lasso_scenario_{}_{}'.format(_scenario, _alpha)  # Identifier string (unused)

    # Grid search values for optimal k selection in mean and variance estimation
    _grida = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])  # k values for mean
    _gridv = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])  # k values for variance

    # Grid of points where CDF will be evaluated
    z = np.linspace(-2, 100, 1000)  # 1000 points from -2 to 100

    # Open CSV file to save results
    with open('cdf_scenario_{}.csv'.format(_scenario), 'w', newline='') as file:
        logging.info(['SCENARIO', 'SIM', 'N', 'P', 'Xi', 'MSE_ECDF1', 'MSE_ECDF2', 'MSE_NCDF1', 'MSE_NCDF2', 'MSE_GAMLSS'])
        writer = csv.writer(file)
        writer.writerow(['SCENARIO', 'SIM', 'N', 'P', 'Xi', 'MSE_ECDF1', 'MSE_ECDF2', 'MSE_NCDF1', 'MSE_NCDF2', 'MSE_GAMLSS'])

        # Loop over sample sizes
        for _n in _n_list:
            # Loop over dimensions
            for _p in _p_list:
                # Monte Carlo repetitions
                for _i in range(_n_sims):

                    # Generate synthetic dataset
                    _x, _y = simulate(_n, _p, _scenario)

                    # Train K-NN models WITH feature selection (LASSO-based)
                    knna1, fa1, ka1, pva1, knnv1, fv1, kv1, pvv1 = initialize_knn(
                        _x, _y, _grida, _gridv, quantile=(1 - _alpha))
                    # knna1: K-NN regressor for mean with feature selection
                    # ka1: optimal k for mean
                    # knnv1: K-NN regressor for variance with feature selection
                    # kv1: optimal k for variance

                    # Train K-NN models WITHOUT feature selection (all variables)
                    knna2, fa2, ka2, pva2, knnv2, fv2, kv2, pvv2 = no_initialize_knn(
                        _x, _y, _grida, _gridv, quantile=(1 - _alpha))

                    # Prepare data for GAMLSS (R package)
                    r_df = r_package.gamlss_data(_x, _y)

                    # Loop over each observation in the dataset
                    for i, xi in enumerate(_x):
                        # Calculate true mean and variance for this observation
                        f0, g0 = sim_scenario(xi, _scenario)

                        # True theoretical CDF: N(f0, g0)
                        distribution_t = norm(loc=f0, scale=math.sqrt(g0))
                        norm_cdf = distribution_t.cdf(z)  # Ground truth

                        # Reshape observation for prediction
                        xi = xi.reshape(1, -1)

                        # METHOD 1: Empirical CDF from K-NN WITH feature selection
                        pecdf1 = knna1.predict_ecdf(xi, z, k=_k, quantile_value=0.1)

                        # METHOD 2: Empirical CDF from K-NN WITHOUT feature selection
                        pecdf2 = knna2.predict_ecdf(xi, z, k=_k, quantile_value=0.1)

                        # METHOD 3: Normal CDF with K-NN estimates (WITH feature selection)
                        # Uses fixed k=100 for both mean and variance
                        distribution_e1 = norm(
                            loc=knna1.predict(xi, k=_k),
                            scale=knnv1.predict(xi, k=_k))
                        norm_cdf_e1 = distribution_e1.cdf(z)

                        # METHOD 4: Normal CDF with K-NN estimates (WITHOUT feature selection)
                        # Uses optimal k values (ka2 for mean, kv2 for variance)
                        distribution_e2 = norm(
                            loc=knna2.predict(xi, ka2),
                            scale=knnv2.predict(xi, kv2))
                        norm_cdf_e2 = distribution_e2.cdf(z)

                        # METHOD 5: GAMLSS CDF from R
                        with localconverter(pandas2ri.converter):
                            pandas_df = pd.DataFrame(pandas2ri.rpy2py(r_df))
                        r_train = r_package.gamlss_train(pandas_df)
                        r_cdf = r_package.gamlss_cdf(xi, r_train, r_df, z)

                        # VISUALIZATION: Plot all 6 CDFs (5 methods + ground truth)
                        plt.plot(z, pecdf1[0], label='ecdf con seleccion')
                        plt.plot(z, pecdf2[0], label='ecdf sin seleccion')
                        plt.plot(z, norm_cdf, label='theoretical')
                        plt.plot(z, norm_cdf_e1, label='knn con seleccion')
                        plt.plot(z, norm_cdf_e2, label='knn sin seleccion')
                        plt.plot(z, r_cdf, label='gamlss cdf')
                        plt.legend()
                        # Save plot to PDF
                        plt.savefig('scenario_{}_{}_{}_{}_{}.pdf'.format(_scenario, _n, _p, _i, i))
                        plt.clf()  # Clear figure for next iteration

                        # EVALUATION: Compute MSE for each method vs. ground truth
                        mse_ecdf1 = skl.mean_squared_error(pecdf1[0], norm_cdf)
                        mse_ecdf2 = skl.mean_squared_error(pecdf2[0], norm_cdf)
                        mse_ncdf1 = skl.mean_squared_error(norm_cdf_e1, norm_cdf)
                        mse_ncdf2 = skl.mean_squared_error(norm_cdf_e2, norm_cdf)
                        mse_gamlss = skl.mean_squared_error(r_cdf, norm_cdf)

                        # Log results to console
                        logging.info(
                            "{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(
                                _scenario, _i, _n, _p, i, mse_ecdf1, mse_ecdf2, mse_ncdf1, mse_ncdf2, mse_gamlss))

                        # Write results to CSV
                        writer.writerow([_scenario, _i, _n, _p, i,
                                         f"{mse_ecdf1:.2f}", f"{mse_ecdf2:.2f}",
                                         f"{mse_ncdf1:.2f}", f"{mse_ncdf2:.2f}",
                                         f"{mse_gamlss:.2f}"])


