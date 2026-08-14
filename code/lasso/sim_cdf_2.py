import math
import argparse
import os
import sys

import logging

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from knnvs import initialize_knn, no_initialize_knn

import matplotlib.pyplot as plt

plt.style.use('default')
pd.set_option("display.precision", 4)

logging.basicConfig(format='%(message)s', level=logging.INFO)

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
        f0 = 5 * np.sum(z[1:3])   # m: X2,X3         Sm={2,3}
        g0 = 1                    # sigma^2: const   Ssigma=empty
    elif scenario == 2:
        f0 = 0                    #                  Sm=empty
        g0 = 5 * z[0]             # sigma^2: X1      Ssigma={1}
    elif scenario == 3:
        f0 = 5 * np.sum(z[1:3])   # m: X2,X3         Sm={2,3}
        g0 = 5 * z[0]             # sigma^2: X1      Ssigma={1}
    elif scenario == 4:
        f0 = 5 * np.sum(z[0:3])   # m: X1,X2,X3      Sm={1,2,3}
        g0 = 1                    # sigma^2: const   Ssigma=empty
    elif scenario == 5:
        f0 = 0                    #                  Sm=empty
        g0 = 5 * np.sum(z[0:3])   # sigma^2: X1,X2,X3   Ssigma={1,2,3}
    elif scenario == 6:
        f0 = 5 * np.sum(z[0:2])   # m: X1,X2         Sm={1,2}
        g0 = 5 * np.sum(z[2:4])   # sigma^2: X3,X4   Ssigma={3,4}
    elif scenario == 7:
        f0 = 5 * np.sum(z[0:3])   # m: X1,X2,X3      Sm={1,2,3}
        g0 = 5 * np.sum(z[1:4])   # sigma^2: X2,X3,X4   Ssigma={2,3,4}
    elif scenario == 8:
        f0 = 5 * np.sum(z[0:8])   # m: X1..X8        Sm={1,...,8}
        g0 = 1                    # sigma^2: const   Ssigma=empty
    elif scenario == 9:
        f0 = 0                    #                  Sm=empty
        g0 = 5 * np.sum(z[0:8])   # sigma^2: X1..X8  Ssigma={1,...,8}
    elif scenario == 10:
        f0 = 5 * np.sum(z[0:5])   # m: X1..X5        Sm={1,2,3,4,5}
        g0 = 5 * np.sum(z[6:8])   # sigma^2: X7,X8   Ssigma={7,8}
    elif scenario == 11:
        # No aparece en la Tabla 1 (no se reporta). Se deja tal cual.
        f0 = 5 * np.sum(z[0:4])   # m: X1..X4        Sm={1,2,3,4}
        g0 = 5 * np.sum(z[5:10])  # sigma^2: X6..X10 Ssigma={6,7,8,9,10}
    elif scenario == 12:
        # DGP de la Figura 1 / Tabla 5 (sec. 5.2)
        f0 = np.exp(3 * z[0] * z[1])   # m: exp(3*X1*X2)  (interaccion)
        g0 = z[2] + z[3]               # sigma^2: X3+X4
    else:
        f0 = 0
        g0 = 0
    return f0, g0


def sim_config(scenario):
    # Figure 1: dim(X) = 10, four relevant predictors (scenario 12 default).
    n0 = [100000]
    p0 = [10]
    return n0, p0


def simulate(n, d, scenario):
    x = np.random.uniform(0, 1, (n, d)).astype('float32')
    eps = np.random.standard_normal(n)
    y = np.zeros(n).astype('float32')
    for j in range(n):
        f0, g0 = sim_scenario(x[j], scenario)
        y[j] = f0 + math.sqrt(g0) * eps[j]
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figure 1 (conditional distribution) plot generator')
    parser.add_argument('-s', '--scenario', default="12", type=int, help='Simulation scenario')
    parser.add_argument('-a', '--alpha', default="0.05", type=float, help='alpha')
    parser.add_argument('-o', '--output-dir', default=".", type=str,
                        help='Directory where the PDF figure(s) will be saved (default: current directory)')
    args = parser.parse_args()

    _scenario = args.scenario
    _alpha = args.alpha
    _output_dir = args.output_dir
    os.makedirs(_output_dir, exist_ok=True)

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

    _n_sims = 1            # One realization for the figure
    _n_list, _p_list = sim_config(_scenario)

    _grida = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    _gridv = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])

    z = np.linspace(-2, 100, 1000)

    # Representative point(s) at which the conditional CDF is drawn (X = z fixed).
    # Widen to e.g. list(range(10)) to generate several and pick the clearest one.
    test_idx = [0]

    for _n in _n_list:
        for _p in _p_list:
            for _i in range(_n_sims):

                _x, _y = simulate(_n, _p, _scenario)
                knna1, fa1, ka1, pva1, knnv1, fv1, kv1, pvv1 = initialize_knn(
                    _x, _y, _grida, _gridv, quantile=(1 - _alpha))
                knna2, fa2, ka2, pva2, knnv2, fv2, kv2, pvv2 = no_initialize_knn(
                    _x, _y, _grida, _gridv, quantile=(1 - _alpha))

                # GAMLSS trained ONCE
                r_df = r_package.gamlss_data(_x, _y)
                with localconverter(pandas2ri.converter):
                    pandas_df = pd.DataFrame(pandas2ri.rpy2py(r_df))
                r_train = r_package.gamlss_train(pandas_df)

                for i in test_idx:
                    xi = _x[i]
                    f0, g0 = sim_scenario(xi, _scenario)
                    distribution_t = norm(loc=f0, scale=math.sqrt(g0))
                    xi = xi.reshape(1, -1)

                    pecdf1 = knna1.predict_ecdf(xi, z, k=ka1, quantile_value=0.1)
                    pecdf2 = knna2.predict_ecdf(xi, z, k=ka2, quantile_value=0.1)
                    norm_cdf = distribution_t.cdf(z)

                    norm_cdf_e1 = norm(
                        loc=knna1.predict(xi, ka1),
                        scale=np.sqrt(np.clip(knnv1.predict(xi, kv1), 1e-12, None))).cdf(z)
                    norm_cdf_e2 = norm(
                        loc=knna2.predict(xi, ka2),
                        scale=np.sqrt(np.clip(knnv2.predict(xi, kv2), 1e-12, None))).cdf(z)

                    r_cdf = r_package.gamlss_cdf(xi, r_train, r_df, z)

                    # ---- Publication figure (matches Figure 1 legend) ----
                    plt.figure(figsize=(10, 8))
                    plt.grid(True, linestyle='-', alpha=0.1)

                    plt.plot(z, pecdf1[0], label='ECDF of kNN with feature selection')
                    plt.plot(z, pecdf2[0], label='ECDF of kNN without feature selection')
                    plt.plot(z, norm_cdf, label='Theoretical value')
                    plt.plot(z, norm_cdf_e1, label='CDF of kNN with feature selection')
                    plt.plot(z, norm_cdf_e2, label='CDF of kNN without feature selection')
                    plt.plot(z, r_cdf, label='CDF of GAMLSS')

                    plt.xlabel('X', fontsize=10)
                    plt.ylabel('Cumulative probability', fontsize=10)
                    legend = plt.legend(frameon=False, fontsize=10)
                    legend.get_frame().set_linewidth(0)
                    legend.get_frame().set_facecolor('none')  # Fully transparent background
                    plt.savefig(os.path.join(
                        _output_dir, 'scenario_{}_{}_{}_{}_{}.pdf'.format(_scenario, _n, _p, _i, i)))
                    plt.close()

                    mse_ecdf1 = skl.mean_squared_error(pecdf1[0], norm_cdf)
                    mse_ecdf2 = skl.mean_squared_error(pecdf2[0], norm_cdf)
                    mse_ncdf1 = skl.mean_squared_error(norm_cdf_e1, norm_cdf)
                    mse_ncdf2 = skl.mean_squared_error(norm_cdf_e2, norm_cdf)
                    mse_gamlss = skl.mean_squared_error(r_cdf, norm_cdf)

                    logging.info(
                        "{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(
                            _scenario, _i, _n, _p, i, mse_ecdf1, mse_ecdf2, mse_ncdf1, mse_ncdf2, mse_gamlss))