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
from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
import multiprocessing as mp
from joblib import Parallel, delayed
import logging
from scipy.stats import ttest_rel
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

# Fold sizes as fractions of n, in the order D1..D4 of Figure 2: feature
# selection for the mean, mean fit + k, feature selection for the variance,
# variance fit + k. They are deliberately unequal. The two selection tasks
# are not equally hard: the mean is selected against Gaussian noise, while
# the variance is selected against the squared-residual pseudo-response
# sigma^2 * chi2_1, whose noise-to-signal ratio is 1.77 and whose kurtosis
# is 15. Measured on the simulation scenarios, mean selection stays at
# perfect accuracy on as little as 6% of n, while variance selection fails
# to reject anything on 25% of it. Moving budget from D1 to D3 leaves the
# fitting folds D2 and D4 untouched, so the estimators are unaffected.
# Theorem 1 only asks that n_j / n -> pi_j in (0, 1), which unequal folds
# satisfy.
FOLD_FRACTIONS = (0.15, 0.25, 0.35, 0.25)

# Candidate k values, as fractions of the fold the model is built on. A
# fixed grid cannot serve every sample size: 2000 neighbours is 160% of the
# fold at n=5000 and 8% of it at n=100000, so the same grid was unusable at
# one end and saturated at the other. Expressed as fractions, the range
# spans the k values the method actually selects (0.002 to 0.4 of the fold
# across the published runs).
K_GRID_FRACTIONS = (0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.12, 0.25, 0.5)
MIN_K = 5


def k_grid(n_train: int) -> npt.NDArray[np.intp]:
    """Candidate k values for a model built on n_train points.

    Returns K_GRID_FRACTIONS scaled by n_train, floored at MIN_K, with
    duplicates removed and values the index cannot honour dropped.
    """
    g = np.round(np.asarray(K_GRID_FRACTIONS, dtype=float) * int(n_train))
    g = np.unique(np.maximum(g, MIN_K).astype(np.intp))
    return g[g <= int(n_train) - 1]
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

    @property
    def n_train(self) -> int:
        """Number of points stored in the FAISS index.

        This is the hard ceiling on k: asking the index for more neighbours
        than it holds makes FAISS pad the result with -1, which numpy would
        silently read as the last element of ``_y``.
        """
        return int(self._x.shape[0])

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
        yp = np.mean(self._y[indices], axis=1)
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
            Accepted and validated for API symmetry with the other predict
            methods, but not used: the eCDF is determined by the neighbours.

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
        # Empirical CDF: fraction of neighbours <= z, for each sample.
        # Sorting each row and binary-searching z gives the same counts as
        # comparing every (neighbour, z) pair, without materialising the
        # (n_samples, k, len(z)) boolean array that comparison would need.
        y_sorted = np.sort(y_neighbors, axis=1)
        counts = np.empty((y_sorted.shape[0], z.shape[0]), dtype=np.intp)
        for i in range(y_sorted.shape[0]):
            counts[i] = np.searchsorted(y_sorted[i], z, side="right")
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

    # Invert the eCDF of res_0 by interpolating over its jump points. The
    # eCDF is evaluated on all of them in one vectorised call rather than
    # one Python-level call per distinct value.
    slope_changes = np.unique(res_0)
    inverted_edf_0 = interp1d(ecdf_0(slope_changes), slope_changes)

    p = np.linspace(P_VALUE_RANGE_MIN, P_VALUE_RANGE_MAX, num=INTEGRATION_POINTS)
    r = 1 - ecdf_1(inverted_edf_0(1 - p) * b - a)
    i1 = simpson(r, p)

    return i1


def no_initialize_knn(x, y, grida=None, gridv=None, quantile=DEFAULT_QUANTILE_THRESHOLD):
    # Same folds as initialize_knn; only the selection step is skipped
    rng = np.random.RandomState(RANDOM_STATE_DEFAULT)
    n = x.shape[0]
    cuts = np.cumsum([int(n * f) for f in FOLD_FRACTIONS])[:-1]
    i1, i2, i3, i4 = np.split(rng.permutation(n), cuts)
    x2, y2 = x[i2], y[i2]
    x4, y4 = x[i4], y[i4]

    features_avg = np.arange(x.shape[1]); pvalues_avg = None
    knna = KnnBag(x2, y2, selected_rows=np.arange(x2.shape[0]), selected_cols=features_avg)
    k_avg = select_best_k(knna, x2, y2, grid=grida)

    features_var = np.arange(x.shape[1]); pvalues_var = None
    resid4 = np.square(y4 - knna.predict(x4, k_avg))
    knnv = KnnBag(x4, resid4, selected_rows=np.arange(x4.shape[0]), selected_cols=features_var)
    k_var = select_best_k_v(knnv, x4, resid4, grid=gridv)

    return (knna, features_avg, k_avg, pvalues_avg,
            knnv, features_var, k_var, pvalues_var)



def initialize_knn(x, y, grida=None, gridv=None, quantile=DEFAULT_QUANTILE_THRESHOLD):
    # Four balanced folds D1..D4 (~n/4 each), one per task (Fig. 2)
    rng = np.random.RandomState(RANDOM_STATE_DEFAULT)
    n = x.shape[0]
    cuts = np.cumsum([int(n * f) for f in FOLD_FRACTIONS])[:-1]
    i1, i2, i3, i4 = np.split(rng.permutation(n), cuts)
    x1, y1 = x[i1], y[i1]   # D1: feature selection, mean
    x2, y2 = x[i2], y[i2]   # D2: fit mean + select k_avg
    x3, y3 = x[i3], y[i3]   # D3: feature selection, variance
    x4, y4 = x[i4], y[i4]   # D4: fit variance + select k_var

    # Conditional mean
    features_avg, pvalues_avg = select_features(x1, y1, grida=grida, quantile=quantile)
    knna = KnnBag(x2, y2, selected_rows=np.arange(x2.shape[0]), selected_cols=features_avg)
    k_avg = select_best_k(knna, x2, y2, grid=grida)

    # Conditional variance (residuals from the mean model)
    resid3 = np.square(y3 - knna.predict(x3, k_avg))
    features_var, pvalues_var = select_features(x3, resid3, grida=gridv, quantile=quantile)
    resid4 = np.square(y4 - knna.predict(x4, k_avg))
    knnv = KnnBag(x4, resid4, selected_rows=np.arange(x4.shape[0]), selected_cols=features_var)
    k_var = select_best_k_v(knnv, x4, resid4, grid=gridv)

    return (knna, features_avg, k_avg, pvalues_avg,
            knnv, features_var, k_var, pvalues_var)


def select_best_k(
    bag: KnnBag,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grid: Optional[npt.NDArray[np.int32]] = None,
) -> int:
    """
    Select the smoothing parameter k by leave-one-out cross-validation.

    For every candidate k in the grid, the k-NN prediction at each point of
    (x, y) is formed from its k nearest neighbours *excluding itself*, and the
    k minimising the resulting mean squared error is returned. This is the
    LOOCV rule of Azadkia (2019) used for both the conditional mean and the
    conditional variance models.

    Parameters
    ----------
    bag : KnnBag
        Model whose index was built on (x, y) — see Notes.
    x : np.ndarray of shape (n_samples, n_features)
        Input features used to build `bag`.
    y : np.ndarray of shape (n_samples,)
        Target values used to build `bag`.
    grid : np.ndarray
        Candidate k values to evaluate.

    Returns
    -------
    int
        The k in the grid minimising the leave-one-out MSE.

    Raises
    ------
    ValueError
        If the grid contains values below 1, or if no candidate k is
        feasible for the number of training points available.

    Notes
    -----
    Candidate k values larger than ``bag.n_train - 1`` are dropped with a
    warning: the index simply does not hold that many neighbours, and asking
    FAISS for them returns -1 padding that numpy would silently read as the
    last element of y. Dropping them also keeps the estimator inside the
    k/n -> 0 regime assumed by the consistency results.

    This function assumes `bag` was built on the same (x, y) passed here, in
    the same row order: row i of x is taken to be point i of the index, and
    that is how the point is recognised and excluded from its own
    prediction. Passing data the bag has not seen would make row numbers
    match unrelated neighbours and silently drop them.

    Neighbours are retrieved once for max(grid) and all candidate k are then
    evaluated from cumulative means over that single retrieval.
    """
    if grid is None:
        grid = k_grid(bag.n_train)
    grid = np.asarray(grid, dtype=np.intp)
    if grid.size == 0:
        raise ValueError("grid must contain at least one candidate k")
    if grid.min() < 1:
        raise ValueError("grid values must be >= 1 for select_best_k")

    # A point cannot have more neighbours than the index holds, minus itself.
    k_ceiling = bag.n_train - 1
    feasible = grid[grid <= k_ceiling]
    if feasible.size == 0:
        raise ValueError(
            f"no candidate k is feasible: the smallest grid value is "
            f"{int(grid.min())} but only {k_ceiling} neighbours are available"
        )
    if feasible.size < grid.size:
        logger.warning(
            "Dropping k values %s: only %d training points are available "
            "(k must be <= %d).",
            grid[grid > k_ceiling].tolist(),
            bag.n_train,
            k_ceiling,
        )
    grid = feasible

    kmax = int(grid.max())
    # One extra neighbour, so that kmax genuine ones remain once the query
    # point itself is removed.
    neighbors = bag.obtain_neighbors(x, kmax + 1)

    # The query point is its own nearest neighbour at distance 0, but FAISS
    # does not reliably return it first. It computes squared distances as
    # |a|^2 + |b|^2 - 2<a,b> in float32, and in low dimension the distance to
    # a genuine neighbour can fall below that expression's rounding error, so
    # several neighbours come back at exactly 0.0 and ties are ordered by
    # index. Dropping column 0 would then keep the point inside its own
    # prediction -- which makes small k look artificially good -- while
    # discarding a real neighbour. So find the point wherever it landed.
    row = np.arange(neighbors.shape[0])
    is_self = neighbors == row[:, None]
    # Column to drop: the query point where present, the last one otherwise.
    drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), kmax)
    keep = np.arange(kmax)[None, :] + (np.arange(kmax)[None, :] >= drop[:, None])
    neighbors = np.take_along_axis(neighbors, keep, axis=1)  # (n_samples, kmax)

    y_neighbors = y[neighbors]
    # cumulative mean over the first m neighbours, m = 1..kmax
    cumsum = np.cumsum(y_neighbors, axis=1)
    counts = np.arange(1, kmax + 1)
    cummean = cumsum / counts[np.newaxis, :]

    # evaluate every candidate k at once
    pred_matrix = cummean[:, grid - 1]  # (n_samples, len(grid))
    errors = np.mean((y[:, None] - pred_matrix) ** 2, axis=0)
    return int(grid[np.argmin(errors)])


def select_best_k_v(
    bag: KnnBag,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grid: npt.NDArray[np.int32],
) -> int:
    """
    Select k for the conditional variance model.

    The paper applies the same LOOCV rule separately to the mean and the
    variance components, so this is `select_best_k` applied to the variance
    bag and its squared-residual targets. It is kept as a separate name
    because the two selections are conceptually distinct steps (folds D2 and
    D4 of Figure 2), not because the criterion differs.

    See `select_best_k` for parameters, return value and caveats.
    """
    return select_best_k(bag, x, y, grid)



def select_features(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grida: Optional[npt.NDArray[np.int32]] = None,
    quantile: float = DEFAULT_QUANTILE_THRESHOLD,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """
    Select relevant features using a kNN-backed paired t-test procedure.

    Behavior summary
    ----------------
    - Splits the provided (x, y) into training/validation subsets.
    - Trains a reference KnnBag on the training subset using all features.
    - For each feature j, trains a KnnBag with feature j removed.
    - On the validation subset computes squared prediction errors with and
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
    # w1 = np.abs(y2 - base_pred)[:, None].repeat(n_features, axis=1)
    w1 = np.square(y2 - base_pred)[:, None].repeat(n_features, axis=1)

    # Precompute selected_cols for each leave-one-out feature to avoid repeated list comprehsion
    feature_indices = np.arange(n_features, dtype=np.intp)
    selected_cols_list = [
        np.delete(feature_indices, j).astype(np.intp) for j in range(n_features)
    ]

    def pred_without_feature_from_cols(selected_cols):
        b = KnnBag(x1, y1, all_rows, selected_cols)
        # return np.abs(y2 - b.predict(x2, k))
        return np.square(y2 - b.predict(x2, k))

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
