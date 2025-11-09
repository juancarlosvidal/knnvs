"""
Variable selection comparison study for heteroscedastic regression.

This module compares the performance of two variable selection methods in
heteroscedastic regression settings:

1. K-NN based variable selection with LASSO initialization (K-NN method)
2. Standard LASSO regression (LASSO method)

For each method, the study evaluates variable selection accuracy and false positive
rate (FPR) for both:
- Variables relevant to the mean function (f0)
- Variables relevant to the variance function (g0)

The simulation generates synthetic data under various scenarios with different
mean and variance structures, trains both methods, and compares their ability
to identify the true relevant variables.

Usage:
    python sim_lasso.py -s 11 -a 0.03

    Arguments:
        -s, --scenario: Simulation scenario number (1-11), default=11
        -a, --alpha: Significance level for variable selection, default=0.03

Output:
    - CSV file: lasso_scenario_{scenario}_{alpha}.csv with accuracy and FPR metrics

Author: Juan
Date: 2025
"""

import math
import argparse

import logging
import csv

import random
import numpy as np
from sklearn import linear_model

from knnvs import initialize_knn

# import matplotlib.pyplot as plt

logging.basicConfig(format='%(message)s', level=logging.INFO)
# logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
# logging.getLogger("matplotlib").setLevel(logging.WARNING)


def sim_scenario(x, scenario):
    """
    Define mean and variance functions for different simulation scenarios.

    This function implements 11 different data-generating processes with varying
    mean and variance structures. Each scenario models heteroscedastic regression
    with Y ~ N(f0(X), g0(X)), where f0 is the mean function and g0 is the variance
    function. This is the vectorized version that operates on matrices.

    Args:
        x (np.ndarray): Covariate matrix of shape (n, p).
        scenario (int): Scenario number (1-11).

    Returns:
        tuple: (f0, g0) where:
            - f0 (np.ndarray): Mean values E[Y|X] of shape (n,)
            - g0 (np.ndarray): Variance values Var[Y|X] of shape (n,)

    Scenarios:
        1: Mean depends on x[1:3], constant variance (homoscedastic)
        2: Constant mean, variance depends on x[0] (pure heteroscedastic)
        3: Mean depends on x[1:3], variance depends on x[0]
        4: Mean depends on x[0:4], constant variance
        5: Constant mean, variance depends on x[0:4]
        6: Mean depends on x[0:3], variance depends on x[3:5]
        7: Mean depends on x[0:4], variance depends on x[1:5]
        8: Mean depends on x[0:8], constant variance (high-dimensional)
        9: Constant mean, variance depends on x[0:8] (high-dimensional)
        10: Mean depends on x[0:6], variance depends on x[7:10]
        11: Mean depends on x[0:4], variance depends on x[5:10]

    Examples:
        >>> x = np.random.uniform(0, 1, (100, 10))
        >>> f0, g0 = sim_scenario(x, scenario=1)
        >>> print(f"Mean shape: {f0.shape}, Variance shape: {g0.shape}")
        Mean shape: (100,), Variance shape: (100,)
    """
    if scenario == 1:
        f0 = np.apply_along_axis(lambda z: 5 * np.sum(z[1:3]), 1, x)
        g0 = np.apply_along_axis(lambda z: 1, 1, x)
    elif scenario == 2:
        f0 = np.apply_along_axis(lambda z: 0, 1, x)
        g0 = np.apply_along_axis(lambda z: 5 * z[0], 1, x)
    elif scenario == 3:
        f0 = np.apply_along_axis(lambda z: 5 * np.sum(z[1:3]), 1, x)
        g0 = np.apply_along_axis(lambda z: 5 * z[0], 1, x)
    elif scenario == 4:
        f0 = np.apply_along_axis(lambda z: 5 * np.sum(z[0:4]), 1, x)
        g0 = np.apply_along_axis(lambda z: 1, 1, x)
    elif scenario == 5:
        f0 = np.apply_along_axis(lambda z: 0, 1, x)
        g0 = np.apply_along_axis(lambda z: 5 * np.sum(z[0:4]), 1, x)
    elif scenario == 6:
        f0 = np.apply_along_axis(lambda z: 5 * np.sum(z[0:3]), 1, x)
        g0 = np.apply_along_axis(lambda z: 5 * np.sum(z[3:5]), 1, x)
    elif scenario == 7:
        f0 = np.apply_along_axis(lambda z: 5 * np.sum(z[0:4]), 1, x)
        g0 = np.apply_along_axis(lambda z: 5 * np.sum(z[1:5]), 1, x)
    elif scenario == 8:
        f0 = np.apply_along_axis(lambda z: 5 * np.sum(z[0:8]), 1, x)
        g0 = np.apply_along_axis(lambda z: 1, 1, x)
    elif scenario == 9:
        f0 = np.apply_along_axis(lambda z: 0, 1, x)
        g0 = np.apply_along_axis(lambda z: 5 * np.sum(z[0:8]), 1, x)
    elif scenario == 10:
        f0 = np.apply_along_axis(lambda z: 5 * np.sum(z[0:6]), 1, x)
        g0 = np.apply_along_axis(lambda z: 5 * np.sum(z[7:10]), 1, x)
    elif scenario == 11:
        f0 = np.apply_along_axis(lambda z: 5 * np.sum(z[0:4]), 1, x)
        g0 = np.apply_along_axis(lambda z: 5 * np.sum(z[5:10]), 1, x)
    else:
        f0 = np.apply_along_axis(lambda z: 0, 1, x)
        g0 = np.apply_along_axis(lambda z: 0, 1, x)
    return f0, g0


def sim_config(scenario):
    """
    Configure sample sizes and dimensions for a given scenario.

    Returns appropriate combinations of sample sizes (n) and number of covariates (p)
    to simulate for each scenario. Different scenarios use different dimensionality
    settings based on their complexity.

    Args:
        scenario (int): Scenario number (1-11).

    Returns:
        tuple: (n0, p0) where:
            - n0 (list): Sample sizes to simulate [5000, 10000, 20000, 50000, 100000]
            - p0 (list): Number of covariates to simulate
                - [5, 10, 20, 25] for scenarios 4-7 (moderate dimensions)
                - [10, 25, 50, 100] for scenarios 8-12 (higher dimensions)
                - [3, 10, 20, 25] for other scenarios (lower dimensions)

    Examples:
        >>> n_list, p_list = sim_config(scenario=1)
        >>> print(f"Sample sizes: {n_list}")
        Sample sizes: [5000, 10000, 20000, 50000, 100000]
    """
    n0, p0 = [5000, 10000, 20000, 50000, 100000], [3, 10, 20, 25]
    # n0, p0 = [20000], [3, 10, 20, 25]  # Alternative for quick testing
    if 4 <= scenario < 8:
        p0 = [5, 10, 20, 25]
    elif 8 <= scenario < 13:
        p0 = [10, 25, 50, 100]
    return n0, p0


def sim_data(n, d, scenario):
    """
    Generate synthetic data for a given scenario.

    Simulates data from a heteroscedastic regression model:
        Y_i = f0(X_i) + sqrt(g0(X_i)) * epsilon_i

    where X_i ~ Uniform(0,1)^d and epsilon_i ~ N(0,1) are independent.
    The functions f0 and g0 are defined by the scenario.

    Args:
        n (int): Sample size (number of observations).
        d (int): Number of covariates/features.
        scenario (int): Scenario number (1-11) determining f0 and g0.

    Returns:
        tuple: (x, y) where:
            - x (np.ndarray): Covariate matrix of shape (n, d), dtype float32
            - y (np.ndarray): Response vector of shape (n,), dtype float64

    Examples:
        >>> np.random.seed(42)
        >>> x, y = sim_data(n=100, d=5, scenario=1)
        >>> print(f"X shape: {x.shape}, Y shape: {y.shape}")
        X shape: (100, 5), Y shape: (100,)
    """
    x = np.random.uniform(0, 1, (n, d)).astype('float32')
    eps = np.random.standard_normal(n)
    f0, g0 = sim_scenario(x, scenario)
    y = f0 + np.sqrt(g0) * eps
    return x, y


def sim_gt(scenario):
    """
    Return ground truth variable indices for a given scenario.

    Identifies which variables are truly relevant for the mean function (f0)
    and which are relevant for the variance function (g0) in each scenario.
    These ground truth sets are used to evaluate variable selection performance.

    Args:
        scenario (int): Scenario number (1-11).

    Returns:
        tuple: (f0, g0) where:
            - f0 (set): Set of variable indices relevant to the mean function
            - g0 (set): Set of variable indices relevant to the variance function

    Scenarios:
        1: f0={1,2}, g0={} - Only mean variables
        2: f0={}, g0={0} - Only variance variable
        3: f0={1,2}, g0={0} - Both mean and variance variables
        4: f0={0,1,2,3}, g0={} - Only mean variables (4 vars)
        5: f0={}, g0={0,1,2,3} - Only variance variables (4 vars)
        6: f0={0,1,2}, g0={3,4} - Disjoint sets
        7: f0={0,1,2,3}, g0={1,2,3,4} - Overlapping sets
        8: f0={0,1,2,3,4,5,6,7}, g0={} - High-dim mean only
        9: f0={}, g0={0,1,2,3,4,5,6,7} - High-dim variance only
        10: f0={0,1,2,3,4,5}, g0={7,8,9} - High-dim disjoint
        11: f0={0,1,2,3}, g0={5,6,7,8,9} - High-dim disjoint

    Examples:
        >>> f_vars, g_vars = sim_gt(scenario=3)
        >>> print(f"Mean variables: {f_vars}")
        Mean variables: {1, 2}
        >>> print(f"Variance variables: {g_vars}")
        Variance variables: {0}
    """
    if scenario == 1:
        f0 = {1, 2}
        g0 = {}
    elif scenario == 2:
        f0 = {}
        g0 = {0}
    elif scenario == 3:
        f0 = {1, 2}
        g0 = {0}
    elif scenario == 4:
        f0 = {0, 1, 2, 3}
        g0 = {}
    elif scenario == 5:
        f0 = {}
        g0 = {0, 1, 2, 3}
    elif scenario == 6:
        f0 = {0, 1, 2}
        g0 = {3, 4}
    elif scenario == 7:
        f0 = {0, 1, 2, 3}
        g0 = {1, 2, 3, 4}
    elif scenario == 8:
        f0 = {0, 1, 2, 3, 4, 5, 6, 7}
        g0 = {}
    elif scenario == 9:
        f0 = {}
        g0 = {0, 1, 2, 3, 4, 5, 6, 7}
    elif scenario == 10:
        f0 = {0, 1, 2, 3, 4, 5}
        g0 = {7, 8, 9}
    elif scenario == 11:
        f0 = {0, 1, 2, 3}
        g0 = {5, 6, 7, 8, 9}
    else:
        f0 = {}
        g0 = {}
    return f0, g0


def assess(ground_truth, selected_vars):
    """
    Evaluate variable selection performance.

    Computes accuracy (recall/sensitivity) and false positive rate (FPR) for
    variable selection methods. Handles edge cases where ground truth or
    selected sets are empty.

    Args:
        ground_truth (set or array-like): Set of indices of true relevant variables.
        selected_vars (set or array-like): Set of indices of variables selected by the method.

    Returns:
        tuple: (accuracy, fpr) where:
            - accuracy (float): Proportion of true variables that were selected
                               = |selected ∩ truth| / |truth|
                               Returns 1 if both sets are empty, 0 if truth is empty but selected is not
            - fpr (float): Proportion of selected variables that are false positives
                          = |selected - truth| / |selected|
                          Returns 0 if no variables were selected

    Examples:
        >>> ground_truth = {0, 1, 2}
        >>> selected_vars = {0, 1, 3, 4}
        >>> acc, fpr = assess(ground_truth, selected_vars)
        >>> print(f"Accuracy: {acc:.2f}, FPR: {fpr:.2f}")
        Accuracy: 0.67, FPR: 0.50

        >>> # Edge case: both empty
        >>> acc, fpr = assess(set(), set())
        >>> print(f"Accuracy: {acc:.2f}, FPR: {fpr:.2f}")
        Accuracy: 1.00, FPR: 0.00
    """
    # Ensure ground_truth is a set
    if not isinstance(ground_truth, set):
        ground_truth = set(ground_truth)

    # Ensure selected_vars is a set
    if not isinstance(selected_vars, set):
        selected_vars = set(selected_vars)

    # Calculate Accuracy (recall/sensitivity)
    if len(ground_truth) == 0 and len(selected_vars) == 0:
        accuracy = 1  # Perfect if both are empty
    else:
        accuracy = 0 if len(ground_truth) == 0 else len(selected_vars.intersection(ground_truth)) / len(ground_truth)

    # Calculate False Positive Rate (FPR)
    fpr = 0 if len(selected_vars) == 0 else len(selected_vars - ground_truth) / len(selected_vars)

    return accuracy, fpr


if __name__ == '__main__':
    """
    Main execution block for variable selection comparison study.

    This simulation study:
    1. Generates synthetic data according to the specified scenario
    2. Performs variable selection using:
       - K-NN method with LASSO initialization (for mean and variance)
       - Standard LASSO regression (only for mean)
    3. Evaluates selection accuracy and FPR against ground truth
    4. Saves results to CSV file

    The simulation loops over:
    - Different sample sizes (n)
    - Different dimensionalities (p)
    - Monte Carlo repetitions (50 by default)

    Metrics reported:
    - ACC_FA_KNN: Accuracy of K-NN method for mean variables
    - FPR_FA_KNN: False positive rate of K-NN method for mean variables
    - ACC_FA_LAS: Accuracy of LASSO method for mean variables
    - FPR_FA_LAS: False positive rate of LASSO method for mean variables
    - ACC_FV_KNN: Accuracy of K-NN method for variance variables
    - FPR_FV_KNN: False positive rate of K-NN method for variance variables
    - ACC_FV_LAS: Accuracy of LASSO method for variance variables (using mean selection)
    - FPR_FV_LAS: False positive rate of LASSO method for variance variables
    """
    parser = argparse.ArgumentParser(
        description='Variable Selection Comparison Study for Heteroscedastic Regression')
    parser.add_argument('-s', '--scenario', default="11", type=int,
                       help='Simulation scenario number (1-11)')
    parser.add_argument('-a', '--alpha', default="0.03", type=float,
                       help='Significance level for variable selection (default: 0.03)')
    args = parser.parse_args()

    _scenario = args.scenario
    _alpha = args.alpha

    # Set random seeds for reproducibility
    random.seed(1)
    np.random.seed(1)

    # Simulation configuration
    _n_sims = 50  # Number of Monte Carlo repetitions
    _n_list, _p_list = sim_config(_scenario)  # Get n and p values to simulate

    _str_sim = 'lasso_scenario_{}_{}'.format(_scenario, _alpha)  # Identifier string

    # Grid search values for optimal k selection in mean and variance estimation
    _grida = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])  # k values for mean
    _gridv = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])  # k values for variance

    # Get ground truth variable sets for this scenario
    ga, gv = sim_gt(_scenario)  # ga: mean vars, gv: variance vars

    # Open CSV file to save results
    with open('lasso_scenario_{}_{}.csv'.format(_scenario, _alpha), 'w', newline='') as file:
        logging.info(['SCENARIO', 'N_SIMS', 'ALPHA', 'N', 'P', 'ACC_FA_KNN', 'FPR_FA_KNN', 'ACC_FA_LAS', 'FPR_FA_LAS',
                      'ACC_FV_KNN', 'FPR_FV_KNN', 'ACC_FV_LAS', 'FPR_FV_LAS'])
        writer = csv.writer(file)
        writer.writerow(['SCENARIO', 'N_SIMS', 'ALPHA', 'N', 'P', 'ACC_FA_KNN', 'FPR_FA_KNN', 'ACC_FA_LAS', 'FPR_FA_LAS',
                         'ACC_FV_KNN', 'FPR_FV_KNN', 'ACC_FV_LAS', 'FPR_FV_LAS'])

        # Loop over sample sizes
        for _n in _n_list:
            # Loop over dimensions
            for _p in _p_list:
                # Monte Carlo repetitions
                for _i in range(_n_sims):
                    # Generate synthetic dataset
                    _x, _y = sim_data(_n, _p, _scenario)

                    # METHOD 1: K-NN based variable selection
                    # Returns selected variables for mean (fa_knn) and variance (fv_knn)
                    knna, fa_knn, ka, pva, knnv, fv_knn, kv, pvv = initialize_knn(
                        _x, _y, _grida, _gridv, quantile=(1 - _alpha))
                    # knna: K-NN regressor for mean
                    # fa_knn: selected variables for mean
                    # knnv: K-NN regressor for variance
                    # fv_knn: selected variables for variance

                    # METHOD 2: Standard LASSO regression
                    clf = linear_model.Lasso(alpha=0.1)
                    clf.fit(_x, _y)
                    indices = np.where(clf.coef_ > 0)  # Variables with non-zero coefficients
                    fa_las = indices[0]  # Selected variables for mean

                    # EVALUATION: Calculate Accuracy and FPR for mean variables
                    acc_fa_knn, fpr_fa_knn = assess(ga, set(fa_knn))  # K-NN method
                    acc_fa_las, fpr_fa_las = assess(ga, set(fa_las))  # LASSO method

                    # EVALUATION: Calculate Accuracy and FPR for variance variables
                    acc_fv_knn, fpr_fv_knn = assess(gv, set(fv_knn))  # K-NN method
                    acc_fv_las, fpr_fv_las = assess(gv, set(fa_las))  # LASSO (using mean selection)

                    # Debug output showing ground truth vs selected variables
                    print('GA {} - FA KNN {} - FA LAS {}'.format(ga, fa_knn, fa_las))
                    print('GV {} - FV KNN {} - FV LAS {}'.format(gv, fv_knn, fa_las))

                    # Log results to console
                    logging.info("{:.0f},{:.0f},{:.2f},{:.0f},{:.0f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},"
                                 "{:.2f}".format(_scenario, _i, _alpha, _n, _p, acc_fa_knn, fpr_fa_knn,
                                                 acc_fa_las, fpr_fa_las, acc_fv_knn, fpr_fv_knn, acc_fv_las,
                                                 fpr_fv_las))

                    # Write results to CSV
                    writer.writerow([_scenario, _i, f"{_alpha:.2f}",  f"{_n:.2f}", f"{_p:.2f}",
                                     f"{acc_fa_knn:.2f}", f"{fpr_fa_knn:.2f}",
                                     f"{acc_fa_las:.2f}", f"{fpr_fa_las:.2f}",
                                     f"{acc_fv_knn:.2f}", f"{fpr_fv_knn:.2f}",
                                     f"{acc_fv_las:.2f}", f"{fpr_fv_las:.2f}"])

                    file.flush()  # Ensure data is written to disk after each iteration
