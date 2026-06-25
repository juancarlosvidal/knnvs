# knnvs

A Python library for k-nearest neighbours (kNN) based conditional mean and variance estimation with automatic feature selection, designed for heteroscedastic regression settings.

## Overview

**knnvs** implements a dual-model kNN approach that separately estimates the conditional mean and conditional variance of a response variable given covariates. The library uses [FAISS](https://github.com/facebookresearch/faiss) for fast nearest-neighbour search and provides statistical feature selection via paired t-tests with Bonferroni correction.

The main contribution is a two-stage pipeline:

1. A **mean model** (`KnnBag`) that predicts E[Y | X] via k-nearest neighbours.
2. A **variance model** (`KnnBag`) trained on squared residuals from the mean model, estimating Var[Y | X].
3. A **ROC-like metric** (`roc`) for comparing two conditional empirical distributions, suitable for group discrimination tasks.

## Core API

| Name                  | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| `KnnBag`            | FAISS-backed kNN model supporting mean and empirical CDF prediction.      |
| `KnnVar`            | Wrapper holding separate `KnnBag` models for mean and variance.         |
| `initialize_knn`    | Train mean and variance models with automatic feature selection.          |
| `no_initialize_knn` | Same pipeline without feature selection (faster baseline).                |
| `select_features`   | Leave-one-out paired t-test feature selection with Bonferroni correction. |
| `select_best_k`     | Data-driven k selection for the mean model (minimises MSE).               |
| `select_best_k_v`   | Data-driven k selection for the variance model.                           |
| `roc`               | ROC-like integral metric comparing two empirical residual distributions.  |

## Quickstart

```python
import numpy as np
from knnvs import KnnVar, initialize_knn, roc

# 1. Prepare data
X_train, X_test = ...   # numpy float32 arrays
y_train, y_test = ...

# 2. Train models with automatic feature and k selection
grida = np.array([5, 10, 20, 50, 100, 200, 500, 1000])  # k grid for mean
gridv = np.array([5, 10, 20, 50, 100, 200, 500, 1000])  # k grid for variance

knna, fa, ka, pva, knnv, fv, kv, pvv = initialize_knn(
    X_train, y_train, grida, gridv, quantile=0.99
)
model = KnnVar(knna, knnv)

# 3. Predict conditional mean and variance
mean_pred = model.predict_average(X_test, k=ka)
var_pred  = model.predict_variance(X_test, k=kv)

# 4. (Optional) Compare two residual distributions
res_0 = y_test - mean_pred_group0
res_1 = y_test - mean_pred_group1
auc = roc(a, b, res_0, res_1)
```

## Repository Structure

```
knnvs/
├── knnvs.py              # Core library
├── environment.yml       # Conda environment
├── sim/                  # Monte Carlo simulation studies
│   ├── scripts/          # Simulation runner scripts (Python + shell)
│   ├── low/              # Low signal-to-noise regime
│   ├── moderate/         # Moderate signal-to-noise regime
│   ├── large/            # Large sample regime
│   └── lasso/            # Comparison with LASSO variable selection
└── real/                 # Real data applications
    ├── nhanes/           # NHANES: waist circumference (diabetic vs. non-diabetic)
    ├── cab/              # CAB dataset analysis
    └── sarcopenia/       # Sarcopenia dataset analysis
```

## Simulations

Simulation studies cover three regimes (low, moderate, large) across varying sample sizes (N = 5 000–100 000) and feature dimensions (p = 3–100). Each scenario runs 300 Monte Carlo replications and reports:

- **MSE** for conditional mean and variance estimation.
- **Feature selection accuracy**: fraction of true relevant variables recovered.
- Results with and without feature selection for comparison.

The `sim/lasso/` study benchmarks knnvs feature selection against LASSO across 11 data-generating scenarios with heteroscedastic noise.

Simulations were originally run on a cluster (Intel Xeon Gold 6248, 20 cores) using `sbatch`:

```bash
nohup ./run.sh sim_1.py
```

Results (CSV files) are stored in the `data/` subfolder of each regime. LaTeX tables can be generated with:

```bash
python sim/scripts/make_table.py --fs lar1.csv --nofs lar1_no.csv
```

## Real Data Applications

| Dataset              | Task                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **NHANES**     | Predict waist circumference; compare diabetic vs. non-diabetic groups via conditional ROC |
| **CAB**        | Conditional mean and variance estimation                                                  |
| **Sarcopenia** | Stratified analysis by gender                                                             |

## Installation

```bash
conda env create -f environment.yml
conda activate knn
```

Key dependencies: `faiss`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `joblib`, `pandas`.
