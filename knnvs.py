"""
kNN-based empirical CDF estimation utilities.

This module implements several variants of k-nearest neighbours (kNN) models
that can predict either the conditional mean or the entire empirical CDF (eCDF)
of a target variable given a feature vector. It includes:

- A variance-aware kNN (KnnVar) that maintains separate models for mean and variance.
- Bagging over multiple FAISS indices for scalability (KnnBag).
- Utilities for model selection: best-k search, feature selection, and ROC evaluation.

The implementations rely on FAISS for fast nearest-neighbour search.

Classes
-------
KnnVar
    Variance-aware kNN with separate models for mean and variance prediction
KnnBag
    kNN implementation using FAISS with feature selection

Functions
---------
select_best_k : Find optimal k value based on MSE
select_best_k_v : Find optimal k value for variance estimation
select_features : Feature selection using kNN approach
initialize_knn : Initialize kNN models with feature selection
no_initialize_knn : Initialize kNN models without feature selection
roc : Compute ROC-like metric for distribution comparison
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
    kNN model with separate prediction for mean and variance.

    This class maintains two independent kNN models: one for predicting
    mean values and another for predicting variances. This allows for
    uncertainty quantification in predictions.

    Parameters
    ----------
    knna : KnnBag
        kNN model for predicting mean values.
    knnv : KnnBag
        kNN model for predicting variances.

    Attributes
    ----------
    _knna : KnnBag
        Internal kNN model for mean prediction.
    _knnv : KnnBag
        Internal kNN model for variance prediction.

    Examples
    --------
    >>> knna = KnnBag(X_train, y_train, rows, cols_mean)
    >>> residuals = np.square(y_train - knna.predict(X_train, k=10))
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

        self._x = x[np.ix_(selected_rows, selected_cols)]
        self._y = y[np.ix_(selected_rows)]
        self._selected_rows = selected_rows
        self._selected_cols = selected_cols
        self._index = faiss.IndexFlatL2(self._x.shape[1])
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

        xt = x[np.ix_(range(x.shape[0]), self._selected_cols)]
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

        xt = x[np.ix_(range(x.shape[0]), self._selected_cols)]
        distances, indices = self._index.search(xt, k=int(k))
        result = []
        for row in self._y[indices]:
            ecdf = ECDF(row)
            result.append(ecdf(z))
        return np.array(result)

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

        xt = x[np.ix_(range(x.shape[0]), self._selected_cols)]
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
    kmax = max(grid)
    # Get kmax+1 neighbors and exclude the point itself (first neighbor)
    neighbors = bag.obtain_neighbors(x, kmax + 1)[:, 1:]

    error = np.zeros((y.shape[0], len(grid)))

    for i in range(y.shape[0]):
        for k_idx in range(len(grid)):
            k_value = grid[k_idx]
            prediction = np.mean(y[neighbors[i, 0:k_value]])
            error[i, k_idx] = float(y[i] - prediction) ** 2

    error = np.mean(error, axis=0)
    min_error_index = np.argmin(error)

    return grid[min_error_index]


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
    kmax = max(grid)
    # Get all kmax+1 neighbors
    neighbors = bag.obtain_neighbors(x, kmax + 1)[:, 0:]

    error = np.zeros((y.shape[0], len(grid)))

    for i in range(y.shape[0]):
        for k_idx in range(len(grid)):
            k_value = grid[k_idx]
            # Start from index 1 to exclude the exact same point
            prediction = np.mean(y[neighbors[i, 1:k_value]])
            error[i, k_idx] = float(y[i] - prediction) ** 2

    error = np.mean(error, axis=0)
    min_error_index = np.argmin(error)

    return grid[min_error_index]


def select_features(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grida: npt.NDArray[np.int32],
    quantile: float = DEFAULT_QUANTILE_THRESHOLD,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """
    Select relevant features using kNN-based statistical testing.

    This function determines which features contribute significantly to
    prediction accuracy by comparing prediction errors with and without
    each feature using paired t-tests.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target values.
    grida : np.ndarray
        Grid of k values to test.
    quantile : float, default=0.99
        Significance threshold (higher = more conservative selection).

    Returns
    -------
    tuple
        - selected : np.ndarray
            Indices of selected features. If no features pass the test,
            all features are returned.
        - pvalues : np.ndarray
            P-values from the paired t-test for each feature.

    Raises
    ------
    TypeError
        If x, y, or grida are not numpy arrays.
    ValueError
        If quantile is not in (0, 1].

    Notes
    -----
    The function uses Bonferroni correction to control for multiple comparisons.
    A warning is logged if no features pass the significance test.
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

    # Create models with each feature removed
    bags = []
    for j in range(x.shape[1]):
        selected_cols = np.array([i for i in range(x1.shape[1]) if i != j])
        bags.append(KnnBag(x1, y1, all_rows, selected_cols))

    # Compute prediction errors
    w1 = np.zeros((x2.shape[0], len(bags)))  # errors with all features
    w2 = np.zeros((x2.shape[0], len(bags)))  # errors without each feature

    for j, b in enumerate(bags):
        w1[:, j] = np.abs(y2 - bag.predict(x2, k))
        w2[:, j] = np.abs(y2 - b.predict(x2, k))

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


