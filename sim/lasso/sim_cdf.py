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

from knn_cdf import initialize_knn, no_initialize_knn

import matplotlib.pyplot as plt


logging.basicConfig(format='%(message)s', level=logging.INFO)
# logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
# logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Store the original R output and message handlers
original_writeconsole = rpy2.rinterface_lib.callbacks.consolewrite_print
original_writeconsole_warnerror = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
original_writeconsole_flush = rpy2.rinterface_lib.callbacks.consoleflush


# Custom function to suppress output
def suppress_r_output(text):
    pass

# Redirect the R console output and messages to the custom suppress function
rpy2.rinterface_lib.callbacks.consolewrite_print = suppress_r_output
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = suppress_r_output
rpy2.rinterface_lib.callbacks.consoleflush = suppress_r_output
rpy2.rinterface_lib.callbacks.consolewrite_message = suppress_r_output


def sim_scenario(z, scenario):
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
    n0, p0 = [5000, 10000, 20000, 50000, 100000], [3, 10, 20, 25]
    if 4 <= scenario < 8:
        p0 = [5, 10, 20, 25]
    else:
        p0 = [10, 25, 50, 100]
    return n0, p0


def simulate(n, d, scenario):
    x = np.random.uniform(0, 1, (n, d)).astype('float32')
    eps = np.random.standard_normal(n)
    y = np.zeros(n).astype('float32')
    for j in range(n):
        f0, g0 = sim_scenario(x[j], scenario)
        y[j] = f0 + math.sqrt(g0) * eps[j]
    return x, y


def assess(ground_truth, selected_vars):
    # Calculate Accuracy
    accuracy = len(selected_vars.intersection(ground_truth)) / len(ground_truth)
    # False Positive Rate (FPR): FP / (FP + TN)
    fpr = len(selected_vars - ground_truth) / len(selected_vars)
    return accuracy, fpr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('-i', '--input_dir', default="./input_new_m", help='Input_directory')
    parser.add_argument('-s', '--scenario', default="12", type=int, help='Simulation scenario')
    parser.add_argument('-a', '--alpha', default="0.05", type=float, help='alpha')
    args = parser.parse_args()

    _scenario = args.scenario
    _alpha = args.alpha

    random.seed(1)
    np.random.seed(1)

    # Enable the automatic conversion between NumPy and R objects
    numpy2ri.activate()
    # Enable the automatic conversion between Pandas and R objects
    pandas2ri.activate()

    # Read the file with the R code snippet
    with open('./script.r', 'r') as f:
        string = f.read()
    # Parse using STAP
    r_package = SignatureTranslatedAnonymousPackage(string, "my_package")


    _n_sims = 1
    _n_list, _p_list = sim_config(_scenario)
    _k = 100

    _str_sim = 'lasso_scenario_{}_{}'.format(_scenario, _alpha)

    _grida = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    _gridv = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])

    z = np.linspace(-2, 100, 1000)

    with open('cdf_scenario_{}.csv'.format(_scenario), 'w', newline='') as file:
        logging.info(['SCENARIO', 'SIM', 'N', 'P', 'Xi', 'MSE_ECDF1', 'MSE_ECDF2', 'MSE_NCDF1', 'MSE_NCDF2', 'MSE_GAMLSS'])
        writer = csv.writer(file)
        writer.writerow(['SCENARIO', 'SIM', 'N', 'P', 'Xi', 'MSE_ECDF1', 'MSE_ECDF2', 'MSE_NCDF1', 'MSE_NCDF2', 'MSE_GAMLSS'])
        for _n in _n_list:
            for _p in _p_list:
                for _i in range(_n_sims):

                    _x, _y = simulate(_n, _p, _scenario)
                    knna1, fa1, ka1, pva1, knnv1, fv1, kv1, pvv1 = initialize_knn(_x, _y, _grida, _gridv, quantile=(1 - _alpha))
                    knna2, fa2, ka2, pva2, knnv2, fv2, kv2, pvv2 = no_initialize_knn(_x, _y, _grida, _gridv, quantile=(1 - _alpha))

                    # Call R to load the data
                    r_df = r_package.gamlss_data(_x, _y)

                    for i, xi in enumerate(_x):
                        f0, g0 = sim_scenario(xi, _scenario)
                        distribution_t = norm(loc=f0, scale=math.sqrt(g0))
                        xi = xi.reshape(1, -1)
                        pecdf1 = knna1.predict_ecdf(xi, z, k=_k, quantile_value=0.1)
                        pecdf2 = knna2.predict_ecdf(xi, z, k=_k, quantile_value=0.1)
                        # Distribución teórica
                        norm_cdf = distribution_t.cdf(z)

                        # Distribución con knn
                        distribution_e1 = norm(loc=knna1.predict(xi, k=_k), scale=knnv1.predict(xi, k=_k))
                        # distribution_e1 = norm(loc=knna1.predict(xi, ka1), scale=knnv1.predict(xi, kv1))
                        norm_cdf_e1 = distribution_e1.cdf(z)

                        distribution_e2 = norm(loc=knna2.predict(xi, ka2), scale=knnv2.predict(xi, kv2))
                        norm_cdf_e2 = distribution_e2.cdf(z)

                        # plt.plot(z, pecdf[0], label='cdf')
                        # plt.plot(z, norm_cdf2, label='cdf') con selección

                        # plt.plot(z, pecdf[0], label='cdf')
                        # plt.plot(z, norm_cdf2, label='cdf') sin selección

                        # Use localconverter to convert the R data frame to a Pandas DataFrame
                        with localconverter(pandas2ri.converter):
                            pandas_df = pd.DataFrame(pandas2ri.rpy2py(r_df))
                        r_train = r_package.gamlss_train(pandas_df)
                        r_cdf = r_package.gamlss_cdf(xi, r_train, r_df, z)


                        plt.plot(z, pecdf1[0], label='ecdf con seleccion')
                        plt.plot(z, pecdf2[0], label='ecdf sin seleccion')
                        plt.plot(z, norm_cdf, label='theoretical')
                        plt.plot(z, norm_cdf_e1, label='knn con seleccion')
                        plt.plot(z, norm_cdf_e2, label='knn sin seleccion')
                        plt.plot(z, r_cdf, label='gamlss cdf')
                        plt.legend()
                        # plt.show()
                        plt.savefig('scenario_{}_{}_{}_{}_{}.pdf'.format(_scenario, _n, _p, _i, i))
                        plt.clf()  # or plt.close()

                        mse_ecdf1 = skl.mean_squared_error(pecdf1[0], norm_cdf)
                        mse_ecdf2 = skl.mean_squared_error(pecdf2[0], norm_cdf)
                        mse_ncdf1 = skl.mean_squared_error(norm_cdf_e1, norm_cdf)
                        mse_ncdf2 = skl.mean_squared_error(norm_cdf_e2, norm_cdf)
                        mse_gamlss = skl.mean_squared_error(r_cdf, norm_cdf)

                        logging.info(
                            "{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(
                                _scenario, _i, _n, _p, i, mse_ecdf1, mse_ecdf2, mse_ncdf1, mse_ncdf2, mse_gamlss))

                        writer.writerow([_scenario, _i, _n, _p, i,
                                         f"{mse_ecdf1:.2f}", f"{mse_ecdf2:.2f}",
                                         f"{mse_ncdf1:.2f}", f"{mse_ncdf2:.2f}",
                                         f"{mse_gamlss:.2f}"])


