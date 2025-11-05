import faiss
import math
# import time

from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
import numpy as np
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
from statsmodels.distributions.empirical_distribution import ECDF


def roc(a, b, res_0, res_1):
    res_0 = res_0.squeeze()
    res_1 = res_1.squeeze()
    ecdf_0 = ECDF(res_0)
    ecdf_1 = ECDF(res_1)

    sample_edf = ecdf_0
    slope_changes = sorted(set(res_0))
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    inverted_edf_0 = interp1d(sample_edf_values_at_slope_changes, slope_changes)

    p = np.linspace(0.001, 0.999, num=100)
    # aux = inverted_edf_0(1-p) * b - a
    # print('Aux: {}'.format(aux))
    r = 1 - ecdf_1(inverted_edf_0(1-p) * b - a)
    i1 = simpson(r, p)
    # plt.plot(p, r)
    # plt.title('AUC: {}'.format(i1))
    # plt.show()
    #
    # print('AUC: {}'.format(i1))
    return i1




def no_initialize_knn(x, y, grida, gridv, quantile=0.99):
    # We split the dataset in 4 parts
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.50, random_state=11)
    x11, x12, y11, y12 = train_test_split(x1, y1, test_size=0.50, random_state=11)
    x21, x22, y21, y22 = train_test_split(x2, y2, test_size=0.50, random_state=11)
    x31, x32, y31, y32 = train_test_split(x22, y22, test_size=0.50, random_state=11)

    # The first split is used to select the relevant features on average
    fa, pva = np.arange(x11.shape[1]), None
    # The second split is used to select the best k for knn average prediction
    knna = KnnBag(x12, y12, selected_rows=np.arange(x12.shape[0]), selected_cols=fa)
    ka = select_best_k(knna, x12, y12, grid=grida)
    # The third split is used to select the best k for knn variance prediction
    r21 = np.square(y21 - knna.predict(x21, ka))
    fv, pvv = np.arange(x21.shape[1]), None
    # The fourth split is used to select the best k for the knn variance prediction
    r31 = np.square(y31 - knna.predict(x31, ka))
    knnv = KnnBag(x31, r31, selected_rows=np.arange(x31.shape[0]), selected_cols=fv)
    r32 = np.square(y32 - knna.predict(x32, ka))
    kv = select_best_k_v(knnv, x32, r32, grid=gridv)

    return knna, fa, ka, pva, knnv, fv, kv, pvv


def initialize_knn(x, y, grida, gridv, quantile=0.99):
    # We split the dataset in 4 parts
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.50, random_state=11)
    x11, x12, y11, y12 = train_test_split(x1, y1, test_size=0.50, random_state=11)
    x21, x22, y21, y22 = train_test_split(x2, y2, test_size=0.50, random_state=11)
    x31, x32, y31, y32 = train_test_split(x22, y22, test_size=0.50, random_state=11)

    # The first split is used to select the relevant features on average
    fa, pva = select_features(x11, y11, grida=grida, quantile=quantile)
    # The second split is used to select the best k for knn average prediction
    knna = KnnBag(x12, y12, selected_rows=np.arange(x12.shape[0]), selected_cols=fa)
    ka = select_best_k(knna, x12, y12, grid=grida)
    # The third split is used to select the best k for knn variance prediction
    r21 = np.square(y21 - knna.predict(x21, ka))
    fv, pvv = select_features(x21, r21, grida=gridv, quantile=quantile)
    # The fourth split is used to select the best k for the knn variance prediction
    r31 = np.square(y31 - knna.predict(x31, ka))
    knnv = KnnBag(x31, r31, selected_rows=np.arange(x31.shape[0]), selected_cols=fv)
    r32 = np.square(y32 - knna.predict(x32, ka))
    kv = select_best_k_v(knnv, x32, r32, grid=gridv)

    return knna, fa, ka, pva, knnv, fv, kv, pvv


def select_best_k(bag, x, y, grid):
    """
    select_best_k is a function designed to determine the optimal value of k in
    a k-nearest neighbors (kNN) context. It does so by evaluating the mean squared
    error (MSE) for different values of k and selecting the one with the minimum
    error. This function is particularly useful in kNN regression or classification
    tasks where the choice of k significantly impacts the model's performance.

    :param bag (KnnBag):
        A knn-bag object that contains a method obtain_neighbors to retrieve the
        nearest neighbors for a given data point.
    :param x (array-like):
        The input features for which the nearest neighbors need to be found.
    :param y (array-like):
        The target values corresponding to the input features.
    :param grid:
        An array or list of integer values of k to be tested.
    :return:
        A tuple containing:
            - grid[min_error_index]: The value of k from grid that resulted in the lowest MSE.
            - error[min_error_index]: The minimum MSE achieved.
    """
    kmax = max(grid)
    neighbors = bag.obtain_neighbors(x, kmax + 1)[:, 1:]

    error = np.zeros((y.shape[0], len(grid)))
    for i in range(y.shape[0]):
        for k in range(len(grid)):
            error[i, k] = float(y[i] - np.mean(y[neighbors[i, 0:grid[k]]])) ** 2
            # error[i, k] = float(y[i] - 1 / grid[k] * np.sum(y[neighbors[i, 0:grid[k]]])) ** 2
    # error = 1 / y.shape[0] * np.sum(error, axis=0)
    error = np.mean(error, axis=0)
    min_error_index = np.argmin(error)
    return grid[min_error_index]


def select_best_k_v(bag, x, y, grid):
    """
    select_best_k is a function designed to determine the optimal value of k in
    a k-nearest neighbors (kNN) context. It does so by evaluating the mean squared
    error (MSE) for different values of k and selecting the one with the minimum
    error. This function is particularly useful in kNN regression or classification
    tasks where the choice of k significantly impacts the model's performance.

    :param bag (KnnBag):
        A knn-bag object that contains a method obtain_neighbors to retrieve the
        nearest neighbors for a given data point.
    :param x (array-like):
        The input features for which the nearest neighbors need to be found.
    :param y (array-like):
        The target values corresponding to the input features.
    :param grid:
        An array or list of integer values of k to be tested.
    :return:
        A tuple containing:
            - grid[min_error_index]: The value of k from grid that resulted in the lowest MSE.
            - error[min_error_index]: The minimum MSE achieved.
    """
    kmax = max(grid)
    neighbors = bag.obtain_neighbors(x, kmax + 1)[:, 0:]

    error_1 = np.zeros((y.shape[0], len(grid)))
    error_2 = np.zeros((y.shape[0], len(grid)))
    error_3 = np.zeros((y.shape[0], len(grid)))
    for i in range(y.shape[0]):
        for k in range(len(grid)):
            error_1[i, k] = float(y[i] - np.mean(y[neighbors[i, 1:grid[k]]])) ** 2
            # error_2[i, k] = float(y[i] - np.mean(y[neighbors[i, 0:grid[k]]])) ** 2
            # error[i, k] = float(np.sqrt(y[i]) - np.sqrt(1 / grid[k] * np.sum(y[neighbors[i, 0:grid[k]]]) ** 2))
    # error_3 = np.abs(error_1 - error_2)
    error = np.mean(error_1, axis=0)
    min_error_index = np.argmin(error)
    return grid[min_error_index]


def select_features(x, y, grida, quantile=0.99):
    """
    select_features_avg is a function designed for feature selection in a
    dataset using a k-nearest neighbors (kNN) approach. The function determines
    the most significant features based on their impact on the prediction
    accuracy of a kNN model. It employs a train-test split, calculates the
    optimal k value, and then assesses the importance of each feature by
    comparing the error reduction when the feature is excluded.

    :param x:
        The input matrix.
    :param y:
        The target array of alues corresponding to the input features.
    :param grida:
        An array or list of integer values of k to be tested for finding the best k.
    :param quantile:
        The quantile used to determine the threshold for feature selection. Default is 0.99.
    :return:
        A tuple containing:
            - An array of indices of the selected features.
            - The optimal k value (k1) found for the full feature set.
    """
    assert type(x) == np.ndarray, "x must be a numpy array"
    assert type(y) == np.ndarray, "y must be a numpy array"
    assert type(grida) == np.ndarray, "grida must be a numpy array"
    assert 0 < quantile <= 1, "quantile must be defined between 0 and 1"

    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.33, random_state=42)

    bag = KnnBag(x1, y1, np.array(range(x1.shape[0])), np.array(range(x1.shape[1])))
    k = select_best_k(bag, x1, y1, grida)

    # We create the -j column bags
    bags = []
    for j in range(x.shape[1]):
        selected_cols = np.array([i for i in range(x1.shape[1]) if i != j])
        bags.append(KnnBag(x1, y1, np.array(range(x1.shape[0])), selected_cols))

    w1 = np.zeros((x2.shape[0], len(bags)))
    w2 = np.zeros((x2.shape[0], len(bags)))
    for j, b in enumerate(bags):
        w1[:, j] = np.abs(y2 - bag.predict(x2, k))
        w2[:, j] = np.abs(y2 - b.predict(x2, k))
    logging.debug("Media w1: {}".format(np.mean(w1, axis=0)))
    logging.debug("Media w2: {}".format(np.mean(w2, axis=0)))

    res = ttest_rel(w1, w2, alternative='less')
    threshold = np.array(res.pvalue) < (1 - quantile) / x.shape[1]

    logging.debug("Thresholds {}".format(threshold))
    logging.debug("Pvalores {}".format(res.pvalue))

    selected = np.argwhere(threshold).flatten()

    if selected.size == 0:
        selected = np.arange(x.shape[1])
    return selected, res.pvalue


###############################################################################
class Knn(object):
    """
    Knn is an abstract base class designed to serve as a blueprint for
    implementing k-nearest neighbors (kNN) models with specific prediction
    behaviors. The class outlines the structure for the key method: predict,
    which is intended to be overridden in derived classes with
    concrete implementation.
    """

    def predict(self, x, k):
        """
        An abstract method intended to be implemented in derived classes. This
        method should provide the logic to predict outcomes based on the average
        of the target values of the nearest neighbors.

        :param k:
            The number of nearest neighbors to consider.
        :param x:
            The input features matrix for which predictions are to be made.
        :return:
            The method is expected to return predictions, typically the average of
            the target values of the k-nearest neighbors.
        :raises:
            NotImplementedError to ensure that this abstract method is overridden
            in derived classes.
        """
        raise NotImplementedError("Abstract method to implement")


###############################################################################
class KnnVar(Knn):

    def __init__(self, knna, knnv):
        self._knna = knna
        self._knnv = knnv

    def predicta(self, x, k=10):
        """
        The predict method in the KnnBag class is an implementation of the
        predict abstract method from the Knn class. It is specifically
        designed to predict outcomes based on the average of the target values
        of the nearest neighbors. This method is part of the k-nearest
        neighbors (kNN) algorithm, and it uses a subset of features (as specified
        in the KnnBag class) for finding the nearest neighbors.

        :param x:
            The input features dataset for which predictions are to be made.
        :param k:
            The number of nearest neighbors to consider in the prediction. This
            value must be greater than or equal to 1.
        :return:
            An array of predicted values, each being the average of the target
            values of the nearest neighbors for the corresponding data point in x.
        """
        assert k > 0, "k must be equal or greater than 1"
        assert type(x) == np.ndarray, "x must be a numpy array"
        yp = self._knna.predict(x, k)
        return yp

    def predictv(self, x, k=10):
        """
        The predict method in the KnnBag class is an implementation of the
        predict abstract method from the Knn class. It is specifically
        designed to predict outcomes based on the average of the target values
        of the nearest neighbors. This method is part of the k-nearest
        neighbors (kNN) algorithm, and it uses a subset of features (as specified
        in the KnnBag class) for finding the nearest neighbors.

        :param x:
            The input features dataset for which predictions are to be made.
        :param k:
            The number of nearest neighbors to consider in the prediction. This
            value must be greater than or equal to 1.
        :return:
            An array of predicted values, each being the average of the target
            values of the nearest neighbors for the corresponding data point in x.
        """
        assert k > 0, "k must be equal or greater than 1"
        assert type(x) == np.ndarray, "x must be a numpy array"
        yp = self._knnv.predict(x, k)
        return yp


###############################################################################
class KnnBag(Knn):
    """
    KnnBag is a subclass of the Knn class, implementing specific functionalities
    for k-nearest neighbors (kNN) prediction using a bagging approach. This class
    extends the kNN methodology by allowing selective use of rows and columns in
    the dataset, and integrates advanced nearest neighbors search using faiss
    library for efficient distance calculations.

    :param x:
        The input features dataset.
    :param y:
        The target values corresponding to the input features.
    :param selected_rows:
        Indices of rows selected for the analysis.
    :param selected_cols:
        Indices of columns (features) selected for the analysis.
    """

    def __init__(self, x, y, selected_rows, selected_cols):
        assert type(x) == np.ndarray, "x must be a numpy array"
        assert type(y) == np.ndarray, "y must be a numpy array"
        assert type(selected_rows) == np.ndarray, "selected_rows must be a numpy array"
        assert type(selected_cols) == np.ndarray, "selected_cols must be a numpy array"
        self._x = x[np.ix_(selected_rows, selected_cols)]
        self._y = y[np.ix_(selected_rows)]
        self._selected_rows = selected_rows
        self._selected_cols = selected_cols

        self._index = faiss.IndexFlatL2(self._x.shape[1])
        self._index.add(self._x)

    def predict(self, x, k=10):
        """
        The predict method in the KnnBag class is an implementation of the
        predict abstract method from the Knn class. It is specifically
        designed to predict outcomes based on the average of the target values
        of the nearest neighbors. This method is part of the k-nearest
        neighbors (kNN) algorithm, and it uses a subset of features (as specified
        in the KnnBag class) for finding the nearest neighbors.

        :param x:
            The input features dataset for which predictions are to be made.
        :param k:
            The number of nearest neighbors to consider in the prediction. This
            value must be greater than or equal to 1.
        :return:
            An array of predicted values, each being the average of the target
            values of the nearest neighbors for the corresponding data point in x.
        """
        assert k > 0, "k must be equal or greater than 1"
        assert type(x) == np.ndarray, "x must be a numpy array"
        xt = x[np.ix_(range(x.shape[0]), self._selected_cols)]
        distances, indices = self._index.search(xt, k=int(k), )
        yp = np.mean(np.array(self._y[indices]), axis=1)
        return yp

    def obtain_neighbors(self, x, k=10):
        """
        The obtain_neighbors method in the KnnBag class is designed to obtain
        the indices of the nearest neighbors for each data point in a given
        dataset x. This method is crucial for k-nearest neighbors (kNN)
        predictions and utilizes the selected features (self._selected_cols) for
        finding the neighbors.

        :param x:
            The dataset for which the nearest neighbors are to be found.
        :param k:
            The number of nearest neighbors to retrieve. This value must be greater
            than or equal to 1.
        :return:
            indices (numpy array): An array of indices of the nearest neighbors for
            each data point in x. The shape of this array is typically
            (number_of_data_points, k).
        """
        assert k > 0, "k must be equal or greater than 1"
        xt = x[np.ix_(range(x.shape[0]), self._selected_cols)]
        distances, indices = self._index.search(xt, k=int(k), )
        return indices


#################################################################################################
class KnnBagging(Knn):
    """
    KnnBagging.py is a class that extends the functionality of the Knn class,
    implementing a k-nearest neighbors (kNN) algorithm with a bagging (bootstrap
    aggregating) approach. This class enhances the robustness and accuracy of
    kNN predictions by creating multiple subsets (bags) of the dataset and
    aggregating their predictions.

    :param x:
        The input features dataset.
    :param y:
        The target values corresponding to the input features.
    :param selected_features:
        Indices of features selected for the analysis.
    :param max_samples:
        The fraction of samples to draw from x to create each bag. Must be in the interval (0.0, 1.0].
    :param n_bags:
        The number of bags (subsets) to create.
    :param n_jobs:
        The number of parallel jobs to run. If 0, it will use all available CPUs.
    """

    def __init__(self, x, y, selected_features, max_samples=1, n_bags=10, n_jobs=0):
        assert 0 < max_samples <= 1, "max_samples should be in the interval (0.0, 1.0]"
        assert n_bags > 0, "n_bags must be a positive integer"
        assert n_jobs >= 0, "n_jobs must be a positive integer or 0 (all available cpus)"
        assert type(x) == np.ndarray, "x must be a numpy array"
        assert type(y) == np.ndarray, "y must be a numpy array"
        assert type(selected_features) == np.ndarray, "selected_features must be a numpy array"
        self._n_bags = n_bags
        self._n_jobs = n_jobs
        if n_jobs == 0 or n_jobs > mp.cpu_count():
            self._n_jobs = mp.cpu_count()
        else:
            self._n_jobs = n_jobs
        self._bags = self.__fit__(x, y, selected_features, max_samples)

    @staticmethod
    def __bag_fit__(x, y, selected_features, n_rows, n_cols):
        # index_r = np.random.choice(x.shape[0], size=n_rows, replace=True)
        index_r = np.random.choice(x.shape[0], size=n_rows, replace=False)
        # index_r = np.random.default_rng().choice(x.shape[0], size=n_rows, replace=True)
        # index_c = np.random.default_rng().choice(x.shape[1], size=n_cols, replace=False)
        # index_r = np.array(range(x.shape[0]))
        # index_c = np.array(range(x.shape[1]))
        if selected_features is None or selected_features == []:
            selected_features = np.array(range(x.shape[1]))
        return KnnBag(x, y, index_r, selected_features)

    def __fit__(self, x, y, selected_features, max_samples):
        n_rows = max(round(x.shape[0] * max_samples), 1)
        # n_rows = x.shape[0]
        # n_cols = int(math.sqrt(x.shape[1]))
        n_cols = x.shape[1]
        with Parallel(n_jobs=self._n_jobs) as parallel_pool:
            delayed_funcs = [delayed(KnnBagging.__bag_fit__)(x, y, selected_features, n_rows, n_cols)
                             for _ in range(self._n_bags)]
            res = parallel_pool(delayed_funcs)
        return res

    @staticmethod
    def __bag_predict__(bag, x, k):
        return bag.predict(x, k)

    def predict(self, x, k):
        """
        The predict method in the KnnBagging.py class provides an implementation
        for predicting outcomes based on the average prediction from multiple
        "bags" (subsets) in a k-nearest neighbors (kNN) bagging model. This
        method aggregates the predictions from each bag to produce a final
        average prediction, enhancing the robustness and stability of the
        kNN predictions.

        :param k:
            The number of nearest neighbors to consider in each bag for making predictions. Must be a positive integer greater than or equal to 1.
        :param x:
            The input features dataset for which predictions are to be made.
        :return:
            An array of average predicted values, aggregated from the predictions of all bags.
        """
        assert k > 0, "k must be a positive integer equal or greater than 1"
        assert self._bags, "Model not fitted"
        assert type(x) == np.ndarray, "selected_features must be a numpy array"
        with Parallel(n_jobs=self._n_jobs) as parallel_pool:
            delayed_funcs = [delayed(KnnBagging.__bag_predict__)(self._bags[i], x, k)
                             for i in range(self._n_bags)]
            res = parallel_pool(delayed_funcs)
        # values = np.transpose(np.array(res).reshape(self._n_bags, x.shape[0]))
        # avg = np.mean(values, axis=1)
        values = np.array(res)
        avg = np.mean(values, axis=0)
        return avg


#################################################################################################
class KnnChunking(Knn):
    """
    Base class for k-nearest neighbors chunking-based algorithm.
    This class is meant for big data files, so the algorithm loads the dataset by chunks.
    For each chunk, a knn bagging is performed.
    The result of the prediction is the average of n_bags predictions.

    :param filename: string.
        The filename containing the data.
    :param selected_features: array.
        The selected features indexes, where each element is between (0,m_features)
    :param chunk_size: integer greater than 0.
        The number of rows per chunk. Default value is 10e4.
    :param max_samples: integer greater than 0.
        The number of rows per chunk. Default value is 1.
    :param n_bags: integer greater than 0.
        The number of bags. Default value is 10.
    :param n_jobs: integer greater or equal to 0.
        The number of jobs that can run in parallel. If n_jobs=0 then use all available cpus will be used.
        Default value is 0.
    """

    def __init__(self, filename, selected_features=None, chunk_size=10e4, max_samples=1, n_bags=10, n_jobs=0):
        assert chunk_size > 0, "the chunk size (number of rows) must be a positive integer"
        assert n_bags > 0, "the sample size must be a positive integer"
        assert n_jobs >= 0, "rhe number of jobs must be a positive integer or 0 (all available cpus)"
        assert type(selected_features) == np.ndarray, "selected_features must be a numpy array"
        logging.info('Creating knn chunking...')
        # self._tmp_dir = tmp_dir
        # if not os.path.exists(self._tmp_dir):
        #     os.makedirs(self._tmp_dir)
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._chunk_list = KnnChunking.__fit__(filename, selected_features, chunk_size, max_samples,
                                               n_bags, n_jobs, self._tmp_dir)

    @staticmethod
    def __chunk_fit__(chunk, selected_features, max_samples, n_bags, n_jobs, tmp):
        """
        The __chunk_fit__ static method in the KnnBagging.py class is designed to
        fit a k-nearest neighbors (kNN) bagging model to a subset (chunk) of
        the data. It is particularly useful for handling large datasets that
        need to be processed in smaller, more manageable pieces. This method
        also includes functionality for persisting the trained model to disk,
        facilitating scalable machine learning workflows.

        :param chunk:
            A pandas DataFrame representing a chunk of the full dataset. The last column is assumed to be the target variable, with the preceding columns as features.
        :param selected_features:
            Indices of features selected for the analysis.
        :param max_samples:
            The fraction of samples to draw from the chunk to create each bag.
        :param n_bags (int):
            The number of bags (subsets) to create for the kNN bagging model.
        :param n_jobs (int):
            The number of parallel jobs to run for training.
        :param tmp:
            A temporary file path or directory used for storing the trained model.
        :return:
            The file path where the trained KnnBagging.py model is saved.
        """
        z = chunk.to_numpy()
        x = z[:, :z.shape[1] - 1].astype('float32')
        y = z[:, z.shape[1] - 1].astype('float32')
        knn = KnnBagging(x, y, selected_features=selected_features, max_samples=max_samples,
                         n_bags=n_bags, n_jobs=n_jobs)
        # We save the knn as an object
        fid = tmp.name + "/" + str(uuid.uuid4())[:8] + ".pkl.xz"
        # fid = 'tmp/'+str(uuid.uuid4())[:8]+'.pkl'
        with open(fid, 'wb') as f:
            pickle.dump(knn, f, pickle.HIGHEST_PROTOCOL)
        return fid

    @staticmethod
    def __fit__(filename, selected_features, chunk_size, max_samples, n_bags, n_jobs, tmp):
        """
        The __fit__ static method in the KnnBagging.py class is a utility function
        designed to fit k-nearest neighbors (kNN) bagging models to chunks of a
        large dataset. This method is particularly useful for handling datasets
        that are too large to load entirely into memory. It reads the dataset
        in chunks, fits a kNN bagging model to each chunk, and saves these
        models to disk.

        :param filename:
            The file path of the dataset in CSV format.
        :param selected_features:
            Indices of features selected for the analysis.
        :param chunk_size:
            The number of rows per chunk to be read from the CSV file.
        :param max_samples:
            The fraction of samples to draw from each chunk to create bags.
        :param n_bags:
            The number of bags (subsets) to create for each chunk's kNN bagging model.
        :param n_jobs:
            The number of parallel jobs to run for training.
        :param tmp:
            A temporary file path or directory used for storing the trained models.
        :return:
            A list of file paths where the trained kNN bagging models for each chunk
            are saved.
        """
        logging.info('Fitting the baggings of each chunk...')
        reader = pd.read_csv(filename, chunksize=chunk_size, header=None)
        # delayed_funcs = [delayed(self.__chunk_fit__)(chunk, selected_features) for chunk in reader]
        # parallel_pool = Parallel(n_jobs=self._n_jobs)
        # res = parallel_pool(delayed_funcs)
        res = []
        for chunk in reader:
            res.append(KnnChunking.__chunk_fit__(chunk, selected_features, max_samples, n_bags, n_jobs, tmp))
        return res

    @staticmethod
    def __chunk_predict__(fid, k, x):
        """
        The __chunk_predict__ static method in the KnnBagging.py class is
        designed for making predictions on a given chunk of data using a
        previously trained and saved k-nearest neighbors (kNN) bagging model.
        This method is particularly useful for handling large datasets that
        have been processed in smaller, more manageable chunks.

        :param fid:
            The file path where the trained kNN bagging model is saved.
        :param k:
            The number of nearest neighbors to consider in each bag for making
            predictions. Must be a positive integer greater than or equal to 1.
        :param x:
            The input features dataset for which predictions are to be made.
        :return:
            An array of predicted values, obtained by averaging the predictions
            from the kNN bags in the loaded model.
        """
        # We load the knn object
        with open(fid, 'rb') as f:
            knn = pickle.load(f)
        res = knn.predict(x, k)
        return res

    def predict(self, x, k):
        """
        The predict method in the KnnBagging.py class provides a way to make
        predictions on a given input dataset x by averaging the predictions from
        multiple trained k-nearest neighbors (kNN) models. Each model corresponds
        to a chunk of the original dataset, and predictions are made on each chunk
        separately.

        :param x:
            The input features dataset for which predictions are to be made.
        :param k:
            The number of nearest neighbors to consider in each bag for making
            predictions. Must be a positive integer greater than or equal to 1.
        :return:
            An array of average predicted values, aggregated from the predictions
            of all chunk models.
        """
        assert k > 0, "k must be a positive integer equal or greater than 1"
        assert type(x) == np.ndarray, "x must be a numpy array"
        # n = len(self._chunk_list)
        # if self._selected_cols is not None:
        #     x = x[np.ix_(np.array(range(x.shape[0])), self._selected_cols)]
        # delayed_funcs = [delayed(self.__chunk_predict__)(knn, x, k) for knn in self._chunk_list]
        # parallel_pool = Parallel(n_jobs=self._n_jobs)
        # res = parallel_pool(delayed_funcs)
        res = []
        for chunk in self._chunk_list:
            res.append(KnnChunking.__chunk_predict__(chunk, x, k))
        # values = np.transpose(np.array(res).reshape(n, x.shape[0]))
        # avg = np.sum(values, axis=1) / n
        values = np.array(res)
        avg = np.mean(values, axis=0)
        return avg

    def predict_2(self, filename, k):
        """
        The predict_2 method in the KnnBagging.py class is designed for making
        predictions on a large dataset that is read in chunks from a CSV file.
        This method is particularly useful when the dataset is too large to fit
        into memory at once. It iterates over chunks of the dataset, makes
        predictions on each chunk, and aggregates the true values and predictions.

        :param filename:
            The file path of the dataset in CSV format.
        :param k:
            The number of nearest neighbors to consider in each bag for making predictions. Must be a positive integer greater than or equal to 1.
        :return:
            A tuple containing:
                - An array of true target values aggregated from all chunks.
                - An array of predicted values aggregated from all chunks.
        """
        reader = pd.read_csv(filename, chunksize=5e3, header=None)
        y_true = []
        y_pred = []
        for chunk in reader:
            z = chunk.to_numpy()
            x = z[:, :z.shape[1] - 1].astype('float32')
            y = z[:, z.shape[1] - 1].astype('float32').reshape(-1, 1)
            p = self.predict(x, k).reshape(-1, 1)
            y_true.extend(y.tolist())
            y_pred.extend(p.tolist())
        # reader = pd.read_csv(filename, header=None)
        # z = reader.to_numpy()
        # x = z[:, :z.shape[1] - 1].astype('float32')
        # y = z[:, z.shape[1] - 1].astype('float32')
        # prediction = self.predict(x, k)
        return np.array(y_true), np.array(y_pred)

    def __del__(self):
        if hasattr(self, '_tmp_dir'):
            self._tmp_dir.cleanup()
        # os.rmdir(self._tmp_dir)
