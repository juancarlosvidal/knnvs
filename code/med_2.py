import math
import numpy as np
import sklearn.metrics as skl

from knnvs import KnnBag, KnnVar, initialize_knn
from sim_utils import resume_csv
import random
import logging
import csv

# import matplotlib.pyplot as plt

logging.basicConfig(format='%(message)s', level=logging.INFO)
# logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
# logging.getLogger("matplotlib").setLevel(logging.WARNING)


# def f0(z): return 10
def f0(z): return 0


# def g0(z): return 1.75 * (z[0] > 0.5) + 0.25 * (z[0] <= 0.5)
def g0(z): return 5 * np.sum(z[0:3])
# def g0(z): return 5 * z[0]
# def g0(z): return 1 * z[1] + 1 * z[2]


def simulate(n, d):
    x = np.random.uniform(0, 1, (n, d)).astype('float32')
    eps = np.random.standard_normal(n)
    y = np.zeros(n).astype('float32')
    for j in range(n):
        y[j] = f0(x[j]) + math.sqrt(g0(x[j])) * eps[j]
    return x, y


if __name__ == '__main__':

    random.seed(1)
    np.random.seed(1)

    n_sims = 300
    n_list = [(5000, 1), (10000, 1), (20000, 1), (50000, 1), (100000, 1)]
#    p_list = [3, 5, 10]
    p_list = [5, 10, 20, 50]

    str_sim = 'med2-fs-[]-[0:3]'
    fa_gt = np.array([])   # Ground truth of important variables in mean
    fv_gt = np.array([0, 1, 2, 3])  # Ground truth of important variables in variance

    grida = None  # None -> k_grid() lo deriva como fracciones del fold correspondiente
    gridv = None

    OUTFILE = 'med2.csv'
    HEADER = ['DESC', 'N_SIMS', 'ACC_FS_A', 'PVA', 'ACC_FS_V', 'PVV', 'N', 'P', 'KA', 'KV', 'MSEA', 'MSEV']
    mode, combos_done, i_done, n_done = resume_csv(OUTFILE, HEADER, n_sims)

    with open(OUTFILE, mode, newline='') as file:
        writer = csv.writer(file)
        if n_done == 0:
            logging.info(",".join(HEADER))
            writer.writerow(HEADER)
        else:
            logging.info("Resuming {}: {} rows already done".format(OUTFILE, n_done))
        combo_idx = 0
        for n, b in n_list:
            for p in p_list:
                if combo_idx < combos_done:
                    combo_idx += 1
                    continue
                i_start = i_done if combo_idx == combos_done else 0
                combo_idx += 1
                for i in range(i_start, n_sims):
                    x, y = simulate(n, p)
                    knna, fa, ka, pva, knnv, fv, kv, pvv = initialize_knn(x, y, grida, gridv, quantile=(1-0.01))
                    knn = KnnVar(knna, knnv)

                    set1, set2 = set(fa), set(fa_gt)
                    acc_sel_m = len(set1.intersection(set2)) / max(fa.shape[0], fa_gt.shape[0])
                    set1, set2 = set(fv), set(fv_gt)
                    acc_sel_v = len(set1.intersection(set2)) / max(fv.shape[0], fv_gt.shape[0])

                    msea, msev = 0, 0
                    msev2 = 0

                    x = np.random.uniform(0, 1, (5000, p)).astype('float32')
                    # for ka in gridv:
                    #     for kv in gridv:

                    # Theoretical mean
                    ma = np.apply_along_axis(f0, 1, x)
                    # Predicted mean
                    pa = knn.predict_average(x, k=ka)
                    msea = skl.mean_squared_error(ma, pa)
                    # kv = 200
                    # Theoretical variance
                    sigma = np.apply_along_axis(g0, 1, x)
                    # Predicted variance
                    pv = knn.predict_variance(x, k=kv)
                    msev = skl.mean_squared_error(sigma, pv)

                    # for k in gridv:
                    #     pv = knn.predict_variance(x, k=k)
                    #     msev3 = skl.mean_squared_error(sigma, pv)
                    #     print('{} {}'.format(k, msev3))
                    #     plt.scatter(sigma, pv)
                    #     plt.show()

                    logging.info("{},{},{},{},{},{},{},{},{},{},{},{}".format(
                        str_sim, i, acc_sel_m, pva, acc_sel_v, pvv, n, p, ka, kv, msea, msev))
                    writer.writerow([str_sim, i, acc_sel_m, pva, acc_sel_v, pvv, n, p, ka, kv, msea, msev])
                    file.flush()

