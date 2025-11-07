"""
kNN-based empirical CDF estimation utilities.

This module implements several k-nearest neighbours (kNN) utilities oriented
to conditional mean and distribution estimation with simple uncertainty
quantification. It is intentionally lightweight and relies on FAISS for
fast nearest-neighbour search.

Key features
- KnnBag: a FAISS-backed, column/row selective kNN "bag" that supports
  mean predictions and empirical CDF estimation from neighbour targets.
- KnnVar: a small wrapper that keeps separate KnnBag models for the
  conditional mean and the conditional variance (or residuals).
- Utilities for model initialization, feature selection and automatic
  selection of the optimal k (select_best_k / select_best_k_v).
- A ROC-like metric that compares two empirical residual distributions.

Practical notes
- FAISS indexes require float32 contiguous arrays: KnnBag converts data to
  float32 and contiguous memory before building/searching the index.
- The select_features routine uses paired t-tests with Bonferroni
  correction; it is designed for interpretability rather than maximal
  predictive power.
- Functions expect numpy arrays as inputs; light input validation is
  performed and informative exceptions are raised on mismatch.

Typical usage (high level)
1. Split data into train/validation folds.
2. Use initialize_knn or no_initialize_knn to obtain trained KnnBag models
   for mean and variance, selected feature indices and chosen k values.
3. Wrap the two KnnBag models with KnnVar to conveniently predict mean,
   variance and associated eCDFs.
4. Use roc(...) to compute the distribution comparison metric.

Exported names of interest
- KnnBag, KnnVar
- initialize_knn, no_initialize_knn
- select_features, select_best_k, select_best_k_v
- roc

"""

import faiss
import math
from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
import uuid
import pickle
import multiprocessing as mp
from joblib import Parallel, delayed
import tempfile
from scipy import stats
import logging
from scipy.stats import wilcoxon, kstest, ttest_rel
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from sklearn.model_selection import train_test_split
from statsmodels.distributions.empirical_distribution import ECDF

# Configure logger for this module
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TEST_SIZE = 0.5
FEATURE_SELECTION_TEST_SIZE = 0.33
RANDOM_STATE_DEFAULT = 11
RANDOM_STATE_FEATURE_SELECTION = 42
INTEGRATION_POINTS = 100
P_VALUE_RANGE_MIN = 0.001
P_VALUE_RANGE_MAX = 0.999
DEFAULT_K = 10
DEFAULT_QUANTILE_VALUE = 0.1
DEFAULT_QUANTILE_THRESHOLD = 0.99


class KnnVar:
    """
    kNN ensemble for conditional mean and conditional variance.

    Purpose
    -------
    Provide a compact interface to predict both a conditional mean and a
    conditional variance (or squared residuals) by keeping two independent
    KnnBag models:
      - _knna: used to predict the conditional mean (or directly the target)
      - _knnv: used to predict variance / squared residuals

    Design notes
    ------------
    - Keeping separate bags for mean and variance avoids biasing variance
      estimation with the same configuration required for the mean.
    - Predictions are computed as simple averages over k nearest neighbours;
      for variance the model is trained on squared residuals.

    Methods (high level)
    --------------------
    - predict_average(x, k): returns mean predictions for rows in x.
    - predict_variance(x, k): returns variance estimates (expect non-negative).
    - predict_*_ecdf(...): return empirical CDF evaluated at provided z points.

    Examples
    --------
    >>> knna = KnnBag(X_train, y_train, rows, cols_mean)
    >>> residuals = (y_train - knna.predict(X_train, k=10))**2
    >>> knnv = KnnBag(X_train, residuals, rows, cols_var)
    >>> model = KnnVar(knna, knnv)
    >>> mean_pred = model.predict_average(X_test, k=10)
    >>> var_pred = model.predict_variance(X_test, k=10)
    """

    def __init__(self, knna: "KnnBag", knnv: "KnnBag") -> None:
        """
        Initialize KnnVar with separate models for mean and variance.

        Parameters
        ----------
        knna : KnnBag
            kNN model for mean prediction.
        knnv : KnnBag
            kNN model for variance prediction.
        """
        self._knna = knna
        self._knnv = knnv

    def _validate_inputs(
        self,
        x: np.ndarray,
        k: int,
        z: Optional[np.ndarray] = None,
        quantile_value: Optional[float] = None,
    ) -> None:
        """
        Validate common inputs for prediction methods.

        Parameters
        ----------
        x : np.ndarray
            Input features array.
        k : int
            Number of neighbors.
        z : np.ndarray, optional
            Target values for eCDF prediction.
        quantile_value : float, optional
            Quantile value for eCDF prediction.

        Raises
        ------
        ValueError
            If k <= 0 or quantile_value not in (0, 1).
        TypeError
            If x or z are not numpy arrays.
        """
        if k <= 0:
            raise ValueError(f"k must be greater than 0, got {k}")

        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, got {type(x).__name__}")

        if z is not None and not isinstance(z, np.ndarray):
            raise TypeError(f"z must be a numpy array, got {type(z).__name__}")

        if quantile_value is not None:
            if not 0 < quantile_value < 1:
                raise ValueError(
                    f"quantile_value must be in (0, 1), got {quantile_value}"
                )

    def predict_average(
        self, x: npt.NDArray[np.float64], k: int = DEFAULT_K
    ) -> npt.NDArray[np.float64]:
        """
        Predict mean values using k-nearest neighbors.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input features for prediction.
        k : int, default=10
            Number of nearest neighbors to consider.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted mean values.

        Raises
        ------
        ValueError
            If k <= 0.
        TypeError
            If x is not a numpy array.
        """
        self._validate_inputs(x, k)
        return self._knna.predict(x, k)

    def predict_average_ecdf(
        self,
        x: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        k: int = DEFAULT_K,
        quantile_value: float = DEFAULT_QUANTILE_VALUE,
    ) -> npt.NDArray[np.float64]:
        """
        Predict empirical CDF using the mean model.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input features.
        z : np.ndarray
            Values at which to evaluate the eCDF.
        k : int, default=10
            Number of nearest neighbors.
        quantile_value : float, default=0.1
            Quantile value between 0 and 1.

        Returns
        -------
        np.ndarray
            Predicted eCDF values.

        Raises
        ------
        ValueError
            If k <= 0 or quantile_value not in (0, 1).
        TypeError
            If x or z are not numpy arrays.
        """
        self._validate_inputs(x, k, z, quantile_value)
        return self._knna.predict_ecdf(x, z, k, quantile_value)

    def predict_variance(
        self, x: npt.NDArray[np.float64], k: int = DEFAULT_K
    ) -> npt.NDArray[np.float64]:
        """
        Predict variance values using k-nearest neighbors.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input features for prediction.
        k : int, default=10
            Number of nearest neighbors to consider.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted variance values.

        Raises
        ------
        ValueError
            If k <= 0.
        TypeError
            If x is not a numpy array.
        """
        self._validate_inputs(x, k)
        return self._knnv.predict(x, k)

    def predict_variance_ecdf(
        self,
        x: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        k: int = DEFAULT_K,
        quantile_value: float = DEFAULT_QUANTILE_VALUE,
    ) -> npt.NDArray[np.float64]:
        """
        Predict empirical CDF using the variance model.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input features.
        z : np.ndarray
            Values at which to evaluate the eCDF.
        k : int, default=10
            Number of nearest neighbors.
        quantile_value : float, default=0.1
            Quantile value between 0 and 1.

        Returns
        -------
        np.ndarray
            Predicted eCDF values.

        Raises
        ------
        ValueError
            If k <= 0 or quantile_value not in (0, 1).
        TypeError
            If x or z are not numpy arrays.
        """
        self._validate_inputs(x, k, z, quantile_value)
        return self._knnv.predict_ecdf(x, z, k, quantile_value)


class KnnBag:
    """
    kNN implementation using FAISS with feature selection (bagging).

    This class implements k-nearest neighbors with selective use of rows
    and columns in the dataset, and integrates FAISS library for efficient
    nearest neighbor search using L2 distance.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, n_features)
        The input features dataset.
    y : np.ndarray of shape (n_samples,)
        The target values corresponding to the input features.
    selected_rows : np.ndarray of shape (n_selected_rows,)
        Indices of rows selected for the analysis.
    selected_cols : np.ndarray of shape (n_selected_features,)
        Indices of columns (features) selected for the analysis.

    Attributes
    ----------
    _x : np.ndarray
        Selected subset of input features.
    _y : np.ndarray
        Selected subset of target values.
    _selected_rows : np.ndarray
        Indices of selected rows.
    _selected_cols : np.ndarray
        Indices of selected columns.
    _index : faiss.Index
        FAISS index for efficient nearest neighbor search.

    Raises
    ------
    TypeError
        If any of the input arrays are not numpy arrays.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        selected_rows: npt.NDArray[np.int32],
        selected_cols: npt.NDArray[np.int32],
    ) -> None:
        """
        Initialize KnnBag with data and feature selection.

        Parameters
        ----------
        x : np.ndarray
            Input features dataset.
        y : np.ndarray
            Target values.
        selected_rows : np.ndarray
            Row indices to use.
        selected_cols : np.ndarray
            Feature indices to use.

        Raises
        ------
        TypeError
            If any input is not a numpy array.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, got {type(x).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array, got {type(y).__name__}")
        if not isinstance(selected_rows, np.ndarray):
            raise TypeError(
                f"selected_rows must be a numpy array, got {type(selected_rows).__name__}"
            )
        if not isinstance(selected_cols, np.ndarray):
            raise TypeError(
                f"selected_cols must be a numpy array, got {type(selected_cols).__name__}"
            )

        # Ensure integer index arrays
        selected_rows = np.asarray(selected_rows, dtype=np.intp)
        selected_cols = np.asarray(selected_cols, dtype=np.intp)

        # Select data and enforce types/contiguity for FAISS (float32)
        if selected_cols.size == 0:
            raise ValueError("selected_cols must contain at least one feature.")
        self._x = np.ascontiguousarray(
            x[np.ix_(selected_rows, selected_cols)].astype(np.float32)
        )
        self._y = np.ascontiguousarray(y[selected_rows])
        self._selected_rows = selected_rows
        self._selected_cols = selected_cols
        # Build FAISS index (expects float32 contiguous)
        self._index = faiss.IndexFlatL2(int(self._x.shape[1]))
        self._index.add(self._x)

    def predict(
        self, x: npt.NDArray[np.float64], k: int = DEFAULT_K
    ) -> npt.NDArray[np.float64]:
        """
        Predict outcomes based on the average of k-nearest neighbors.

        Uses FAISS for efficient nearest neighbor search and returns the
        mean of the target values of the k nearest neighbors.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input features for prediction.
        k : int, default=10
            Number of nearest neighbors to consider.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted values (mean of k-nearest neighbors).

        Raises
        ------
        ValueError
            If k <= 0.
        TypeError
            If x is not a numpy array.
        """
        if k <= 0:
            raise ValueError(f"k must be greater than 0, got {k}")
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, got {type(x).__name__}")

        # Prepare query matrix (must be float32 contiguous)
        xt = np.ascontiguousarray(x[:, self._selected_cols].astype(np.float32))
        distances, indices = self._index.search(xt, k=int(k))
        yp = np.mean(np.array(self._y[indices]), axis=1)
        return yp

    def predict_ecdf(
        self,
        x: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        k: int = DEFAULT_K,
        quantile_value: float = DEFAULT_QUANTILE_VALUE,
    ) -> npt.NDArray[np.float64]:
        """
        Predict empirical cumulative distribution function.

        For each sample, finds k nearest neighbors and computes the
        empirical CDF of their target values, evaluated at points z.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input features.
        z : np.ndarray
            Values at which to evaluate the eCDF.
        k : int, default=10
            Number of nearest neighbors.
        quantile_value : float, default=0.1
            Quantile value (currently not used in computation).

        Returns
        -------
        np.ndarray of shape (n_samples, len(z))
            eCDF values for each sample.

        Raises
        ------
        ValueError
            If k <= 0 or quantile_value not in (0, 1).
        TypeError
            If x or z are not numpy arrays.
        """
        if k <= 0:
            raise ValueError(f"k must be greater than 0, got {k}")
        if not 0 < quantile_value < 1:
            raise ValueError(f"quantile_value must be in (0, 1), got {quantile_value}")
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, got {type(x).__name__}")
        if not isinstance(z, np.ndarray):
            raise TypeError(f"z must be a numpy array, got {type(z).__name__}")

        xt = np.ascontiguousarray(x[:, self._selected_cols].astype(np.float32))
        distances, indices = self._index.search(xt, k=int(k))

        # y_neighbors: (n_samples, k)
        y_neighbors = self._y[indices]
        z = np.asarray(z)
        # vectorized empirical CDF: fraction of neighbors <= z for each sample
        # result shape -> (n_samples, len(z))
        counts = np.sum(y_neighbors[:, :, None] <= z[None, None, :], axis=1)
        result = counts.astype(float) / float(y_neighbors.shape[1])
        return result

    def obtain_neighbors(
        self, x: npt.NDArray[np.float64], k: int = DEFAULT_K
    ) -> npt.NDArray[np.int32]:
        """
        Obtain indices of k-nearest neighbors.

        This method is useful for manual inspection of neighbors or for
        implementing custom prediction logic.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Dataset for which to find nearest neighbors.
        k : int, default=10
            Number of nearest neighbors to retrieve.

        Returns
        -------
        np.ndarray of shape (n_samples, k)
            Indices of the k-nearest neighbors for each sample.

        Raises
        ------
        ValueError
            If k <= 0.
        """
        if k <= 0:
            raise ValueError(f"k must be greater than 0, got {k}")

        xt = np.ascontiguousarray(x[:, self._selected_cols].astype(np.float32))
        distances, indices = self._index.search(xt, k=int(k))
        return indices


def roc(
    a: float, b: float, res_0: npt.NDArray[np.float64], res_1: npt.NDArray[np.float64]
) -> float:
    """
    Compute ROC-like metric comparing two empirical distributions.

    This function compares two sets of residuals by computing an integral
    that measures the similarity between their empirical CDFs after applying
    an affine transformation.

    Parameters
    ----------
    a : float
        Additive transformation parameter.
    b : float
        Multiplicative transformation parameter.
    res_0 : np.ndarray
        First set of residuals.
    res_1 : np.ndarray
        Second set of residuals.

    Returns
    -------
    float
        Integrated metric value (higher values indicate better alignment).

    Notes
    -----
    The function transforms res_0 by (inverted_edf_0(1-p) * b - a) and
    compares with res_1's eCDF.
    """
    res_0 = res_0.squeeze()
    res_1 = res_1.squeeze()

    ecdf_0 = ECDF(res_0)
    ecdf_1 = ECDF(res_1)

    sample_edf = ecdf_0
    slope_changes = sorted(set(res_0))
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    inverted_edf_0 = interp1d(sample_edf_values_at_slope_changes, slope_changes)

    p = np.linspace(P_VALUE_RANGE_MIN, P_VALUE_RANGE_MAX, num=INTEGRATION_POINTS)
    r = 1 - ecdf_1(inverted_edf_0(1 - p) * b - a)
    i1 = simpson(r, p)

    return i1


def no_initialize_knn(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grida: npt.NDArray[np.int32],
    gridv: npt.NDArray[np.int32],
    quantile: float = DEFAULT_QUANTILE_THRESHOLD,
) -> Tuple[
    KnnBag, npt.NDArray[np.int32], int, None, KnnBag, npt.NDArray[np.int32], int, None
]:
    """
    Initialize kNN models without feature selection.

    Creates two kNN models (for mean and variance) using all available
    features. This is faster than initialize_knn but may be less accurate
    if irrelevant features are present.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target values.
    grida : np.ndarray
        Grid of k values to test for the mean model.
    gridv : np.ndarray
        Grid of k values to test for the variance model.
    quantile : float, default=0.99
        Quantile threshold (not used in this function, kept for API consistency).

    Returns
    -------
    tuple
        - knna : KnnBag
            Trained kNN model for mean prediction
        - features_avg : np.ndarray
            Feature indices used (all features)
        - k_avg : int
            Optimal k for mean model
        - pvalues_avg : None
            No p-values (no feature selection performed)
        - knnv : KnnBag
            Trained kNN model for variance prediction
        - features_var : np.ndarray
            Feature indices used (all features)
        - k_var : int
            Optimal k for variance model
        - pvalues_var : None
            No p-values (no feature selection performed)

    Notes
    -----
    The data is split multiple times to create training/validation sets
    for model selection and evaluation.
    """
    # First split: separate into two halves
    x1, x2, y1, y2 = train_test_split(
        x, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE_DEFAULT
    )

    # Split first half for mean model training
    x11, x12, y11, y12 = train_test_split(
        x1, y1, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE_DEFAULT
    )

    # Split second half for variance model training
    x21, x22, y21, y22 = train_test_split(
        x2, y2, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE_DEFAULT
    )

    # Split for final variance model evaluation
    x31, x32, y31, y32 = train_test_split(
        x22, y22, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE_DEFAULT
    )

    # Use all features (no feature selection)
    features_avg = np.arange(x11.shape[1])
    pvalues_avg = None

    # Train mean model and select optimal k
    knna = KnnBag(
        x12, y12, selected_rows=np.arange(x12.shape[0]), selected_cols=features_avg
    )
    k_avg = select_best_k(knna, x12, y12, grid=grida)

    # Compute residuals for variance model
    residuals_21 = np.square(y21 - knna.predict(x21, k_avg))

    features_var = np.arange(x21.shape[1])
    pvalues_var = None

    residuals_31 = np.square(y31 - knna.predict(x31, k_avg))

    # Train variance model
    knnv = KnnBag(
        x31,
        residuals_31,
        selected_rows=np.arange(x31.shape[0]),
        selected_cols=features_var,
    )

    residuals_32 = np.square(y32 - knna.predict(x32, k_avg))
    k_var = select_best_k_v(knnv, x32, residuals_32, grid=gridv)

    return (
        knna,
        features_avg,
        k_avg,
        pvalues_avg,
        knnv,
        features_var,
        k_var,
        pvalues_var,
    )


def initialize_knn(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grida: npt.NDArray[np.int32],
    gridv: npt.NDArray[np.int32],
    quantile: float = DEFAULT_QUANTILE_THRESHOLD,
) -> Tuple[
    KnnBag,
    npt.NDArray[np.int32],
    int,
    npt.NDArray[np.float64],
    KnnBag,
    npt.NDArray[np.int32],
    int,
    npt.NDArray[np.float64],
]:
    """
    Initialize kNN models with automatic feature selection.

    Creates two kNN models (for mean and variance) with automatic feature
    selection based on statistical significance tests. More computationally
    expensive than no_initialize_knn but typically more accurate.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target values.
    grida : np.ndarray
        Grid of k values to test for the mean model.
    gridv : np.ndarray
        Grid of k values to test for the variance model.
    quantile : float, default=0.99
        Quantile threshold for feature selection (higher = more strict).

    Returns
    -------
    tuple
        - knna : KnnBag
            Trained kNN model for mean prediction
        - features_avg : np.ndarray
            Selected feature indices for mean model
        - k_avg : int
            Optimal k for mean model
        - pvalues_avg : np.ndarray
            P-values from feature selection for mean model
        - knnv : KnnBag
            Trained kNN model for variance prediction
        - features_var : np.ndarray
            Selected feature indices for variance model
        - k_var : int
            Optimal k for variance model
        - pvalues_var : np.ndarray
            P-values from feature selection for variance model

    Notes
    -----
    The function uses multiple train-test splits to avoid overfitting
    during feature selection and model tuning.
    """
    # First split: separate into two halves
    x1, x2, y1, y2 = train_test_split(
        x, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE_DEFAULT
    )

    # Split first half for mean model training
    x11, x12, y11, y12 = train_test_split(
        x1, y1, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE_DEFAULT
    )

    # Split second half for variance model training
    x21, x22, y21, y22 = train_test_split(
        x2, y2, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE_DEFAULT
    )

    # Split for final variance model evaluation
    x31, x32, y31, y32 = train_test_split(
        x22, y22, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE_DEFAULT
    )

    # Perform feature selection for mean model
    features_avg, pvalues_avg = select_features(
        x11, y11, grida=grida, quantile=quantile
    )

    # Train mean model with selected features
    knna = KnnBag(
        x12, y12, selected_rows=np.arange(x12.shape[0]), selected_cols=features_avg
    )
    k_avg = select_best_k(knna, x12, y12, grid=grida)

    # Compute residuals for variance model
    residuals_21 = np.square(y21 - knna.predict(x21, k_avg))

    # Perform feature selection for variance model
    features_var, pvalues_var = select_features(
        x21, residuals_21, grida=gridv, quantile=quantile
    )

    residuals_31 = np.square(y31 - knna.predict(x31, k_avg))

    # Train variance model with selected features
    knnv = KnnBag(
        x31,
        residuals_31,
        selected_rows=np.arange(x31.shape[0]),
        selected_cols=features_var,
    )

    residuals_32 = np.square(y32 - knna.predict(x32, k_avg))
    k_var = select_best_k_v(knnv, x32, residuals_32, grid=gridv)

    return (
        knna,
        features_avg,
        k_avg,
        pvalues_avg,
        knnv,
        features_var,
        k_var,
        pvalues_var,
    )


def select_best_k(
    bag: KnnBag,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grid: npt.NDArray[np.int32],
) -> int:
    """
    Select optimal k value based on mean squared error.

    Evaluates different values of k on validation data and returns the k
    that minimizes the mean squared error.

    Parameters
    ----------
    bag : KnnBag
        Trained kNN model with obtain_neighbors method.
    x : np.ndarray of shape (n_samples, n_features)
        Input features for validation.
    y : np.ndarray of shape (n_samples,)
        Target values for validation.
    grid : np.ndarray
        Array of k values to test.

    Returns
    -------
    int
        Optimal k value from the grid that minimizes MSE.

    Notes
    -----
    This function is computationally efficient as it retrieves neighbors
    only once (for max(grid)) and then evaluates different k values by
    using subsets of those neighbors.
    """
    grid = np.asarray(grid, dtype=np.intp)
    if grid.min() < 1:
        raise ValueError("grid values must be >= 1 for select_best_k")

    kmax = int(grid.max())
    # neighbors_full includes the point itself as first column -> drop it
    neighbors = bag.obtain_neighbors(x, kmax + 1)[:, 1:]  # shape (n_samples, kmax)

    # y_neighbors shape (n_samples, kmax)
    y_neighbors = y[neighbors]
    # cumulative mean of first m neighbors (1..kmax)
    cumsum = np.cumsum(y_neighbors, axis=1)
    counts = np.arange(1, kmax + 1)
    cummean = cumsum / counts[np.newaxis, :]

    # For each candidate k, pick cummean[:, k-1] and compute MSE
    # vectorized evaluation for all candidate k
    grid = np.asarray(grid, dtype=np.intp)
    grid_idx = grid - 1  # zero-based indices into cummean
    pred_matrix = cummean[:, grid_idx]  # shape (n_samples, len(grid))
    errors = np.mean((y[:, None] - pred_matrix) ** 2, axis=0)
    return int(grid[np.argmin(errors)])


def select_best_k_v(
    bag: KnnBag,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grid: npt.NDArray[np.int32],
) -> int:
    """
    Select optimal k value for variance estimation.

    Similar to select_best_k but optimized for variance/residual prediction.
    Includes all neighbors (including the point itself) in the search.

    Parameters
    ----------
    bag : KnnBag
        Trained kNN model with obtain_neighbors method.
    x : np.ndarray of shape (n_samples, n_features)
        Input features for validation.
    y : np.ndarray of shape (n_samples,)
        Target values (typically residuals) for validation.
    grid : np.ndarray
        Array of k values to test.

    Returns
    -------
    int
        Optimal k value from the grid that minimizes MSE.

    Notes
    -----
    Unlike select_best_k, this function includes the point itself in
    neighbor consideration (starts from index 1 instead of 0).
    """
    grid = np.asarray(grid, dtype=np.intp)
    if grid.min() < 1:
        raise ValueError(
            "grid values must be >= 1 for select_best_k_v (at least 1 neighbor excluding self)"
        )

    kmax = int(grid.max())
    # Request kmax+1 neighbors so we can exclude the self-hit at col 0
    neighbors_full = bag.obtain_neighbors(x, kmax + 1)  # shape (n_samples, kmax+1)
    # Exclude self (first column) -> shape (n_samples, kmax)
    neighbors_excl_self = neighbors_full[:, 1 : (kmax + 1)]

    y_neighbors = y[neighbors_excl_self]  # shape (n_samples, kmax)
    cumsum = np.cumsum(y_neighbors, axis=1)
    counts = np.arange(1, kmax + 1)
    cummean = cumsum / counts[np.newaxis, :]

    grid = np.asarray(grid, dtype=np.intp)
    grid_idx = grid - 1
    pred_matrix = cummean[:, grid_idx]
    errors = np.mean((y[:, None] - pred_matrix) ** 2, axis=0)
    return int(grid[np.argmin(errors)])


def select_features(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grida: npt.NDArray[np.int32],
    quantile: float = DEFAULT_QUANTILE_THRESHOLD,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """
    Select relevant features using a kNN-backed paired t-test procedure.

    Behavior summary
    ----------------
    - Splits the provided (x, y) into training/validation subsets.
    - Trains a reference KnnBag on the training subset using all features.
    - For each feature j, trains a KnnBag with feature j removed.
    - On the validation subset computes absolute prediction errors with and
      without each feature and runs a paired t-test (one-sided: errors with
      all features < errors with feature removed).
    - Applies a Bonferroni correction for multiple comparisons.
    - Returns indices of features that pass the corrected significance level
      together with the raw p-values for inspection.

    Returns
    -------
    selected : np.ndarray (shape=(m_selected,))
        Indices of selected features. If no feature passes the threshold the
        function returns all features and logs a warning.
    pvalues : np.ndarray (shape=(n_features,))
        Raw p-values from the paired t-tests (before any ordering).

    Important details
    -----------------
    - The test is one-sided (alternative='less' in ttest_rel): a small
      p-value indicates that removing the feature increases error.
    - The method is intended as a fast, interpretable filter; consider more
      advanced selection (cross-validated wrappers) for production use.
    - Computational complexity: O(n_features * cost_of_training_a_KnnBag).
      On large feature sets or large n_samples consider parallelizing bag
      construction or using no_initialize_knn as a baseline.

    Errors and validation
    ---------------------
    - Raises TypeError if inputs are not numpy arrays.
    - Raises ValueError if quantile is not in (0, 1].

    Example
    -------
    >>> selected, pvals = select_features(X_train, y_train, grida=np.array([5,10,20]))
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy array, got {type(x).__name__}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a numpy array, got {type(y).__name__}")
    if not isinstance(grida, np.ndarray):
        raise TypeError(f"grida must be a numpy array, got {type(grida).__name__}")
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")

    # Split data for feature selection
    x1, x2, y1, y2 = train_test_split(
        x,
        y,
        test_size=FEATURE_SELECTION_TEST_SIZE,
        random_state=RANDOM_STATE_FEATURE_SELECTION,
    )

    # Train model with all features
    all_rows = np.array(range(x1.shape[0]))
    all_cols = np.array(range(x1.shape[1]))
    bag = KnnBag(x1, y1, all_rows, all_cols)
    k = select_best_k(bag, x1, y1, grida)

    # Compute base prediction once and reuse for w1
    n_features = x.shape[1]
    base_pred = bag.predict(x2, k)
    w1 = np.abs(y2 - base_pred)[:, None].repeat(n_features, axis=1)

    # Precompute selected_cols for each leave-one-out feature to avoid repeated list comprehsion
    feature_indices = np.arange(n_features, dtype=np.intp)
    selected_cols_list = [
        np.delete(feature_indices, j).astype(np.intp) for j in range(n_features)
    ]

    def pred_without_feature_from_cols(selected_cols):
        b = KnnBag(x1, y1, all_rows, selected_cols)
        return np.abs(y2 - b.predict(x2, k))

    # Use threads to avoid pickling FAISS-heavy objects (and reduce overhead)
    n_jobs = min(n_features, max(1, mp.cpu_count()))
    w2_cols = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(pred_without_feature_from_cols)(cols) for cols in selected_cols_list
    )
    w2 = np.column_stack(w2_cols)

    logger.debug(
        "Feature selection - Mean error with all features: %s", np.mean(w1, axis=0)
    )
    logger.debug(
        "Feature selection - Mean error without each feature: %s", np.mean(w2, axis=0)
    )

    # Perform paired t-test (one-sided: w1 < w2 means feature is useful)
    res = ttest_rel(w1, w2, alternative="less")

    # Apply Bonferroni correction
    threshold = np.array(res.pvalue) < (1 - quantile) / x.shape[1]

    logger.debug("Feature selection - Significance threshold: %s", threshold)
    logger.debug("Feature selection - P-values: %s", res.pvalue)

    selected = np.argwhere(threshold).flatten()

    # If no features are significant, use all features
    if selected.size == 0:
        logger.warning(
            "No features passed significance test at quantile=%.2f. Using all %d features.",
            quantile,
            x.shape[1],
        )
        selected = np.arange(x.shape[1])

    return (selected, res.pvalue)
