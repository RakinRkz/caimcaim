"""
CAIM
=====

# CAIM (class-attribute interdependence maximization) algorithm for
        supervised discretization

.. note::
    "L. A. Kurgan and K. J. Cios (2004), CAIM discretization algorithm in
    IEEE Transactions on Knowledge and Data Engineering, vol. 16, no. 2, pp. 145-153, Feb. 2004.
    doi: 10.1109/TKDE.2004.1269594"
    .. _a link: http://ieeexplore.ieee.org/document/1269594/

.. module:: caimcaim
   :platform: Unix, Windows
   :synopsis: A simple, but effective discretization algorithm

"""


import cupy as cp
import cudf
from sklearn.base import BaseEstimator, TransformerMixin


class CAIMD(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features='auto'):
        if isinstance(categorical_features, str):
            self._features = categorical_features
            self.categorical = None
        elif (isinstance(categorical_features, list)) or (isinstance(categorical_features, np.ndarray)):
            self._features = None
            self.categorical = categorical_features
        else:
            raise CategoricalParamException(
                "Wrong type for 'categorical_features'. Expected 'auto', an array of indices or labels.")

    def fit(self, X, y):
        self.split_scheme = dict()
        if isinstance(X, pd.DataFrame):
            if isinstance(self._features, list):
                self.categorical = [X.columns.get_loc(label) for label in self._features]
            X = cudf.DataFrame.from_pandas(X)
            y = cp.asarray(y)
        if self._features == 'auto':
            self.categorical = self.check_categorical(X, y)
        categorical = self.categorical

        min_splits = cp.unique(y).size

        for j in range(X.shape[1]):
            if j in categorical:
                continue
            xj = X.iloc[:, j]
            xj = xj.dropna().values
            new_index = cp.argsort(xj)
            xj = xj[new_index]
            yj = y[new_index]
            allsplits = cp.unique(xj)[1:-1].tolist()

            global_caim = -1
            mainscheme = [xj[0], xj[-1]]
            best_caim = 0
            k = 1
            while (k <= min_splits) or ((global_caim < best_caim) and (len(allsplits) > 0)):
                split_points = cp.random.permutation(cp.array(allsplits)).tolist()
                best_scheme = None
                best_point = None
                best_caim = 0
                k = k + 1
                while split_points:
                    scheme = mainscheme[:]
                    sp = split_points.pop()
                    scheme.append(sp)
                    scheme.sort()
                    c = self.get_caim(scheme, xj, yj)
                    if c > best_caim:
                        best_caim = c
                        best_scheme = scheme
                        best_point = sp
                if (k <= min_splits) or (best_caim > global_caim):
                    mainscheme = best_scheme
                    global_caim = best_caim
                    try:
                        allsplits.remove(best_point)
                    except ValueError:
                        raise NotEnoughPoints('The feature #' + str(j) + ' does not have' +
                                              ' enough unique values for discretization!' +
                                              ' Add it to categorical list!')

            self.split_scheme[j] = mainscheme
            print('#', j, ' GLOBAL CAIM ', global_caim)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = cudf.DataFrame.from_pandas(X)
        X_di = X.copy()
        categorical = self.categorical

        scheme = self.split_scheme
        for j in range(X.shape[1]):
            if j in categorical:
                continue
            sh = scheme[j]
            sh[-1] = sh[-1] + 1
            xj = X.iloc[:, j]
            for i in range(len(sh) - 1):
                ind = cp.where((xj >= sh[i]) & (xj < sh[i + 1]))[0]
                X_di.iloc[ind, j] = i
        return X_di.to_pandas()

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_caim(self, scheme, xi, y):
        sp = self.index_from_scheme(scheme[1:-1], xi)
        sp.insert(0, 0)
        sp.append(xi.size)
        n = len(sp) - 1
        isum = 0
        for j in range(n):
            init = sp[j]
            fin = sp[j + 1]
            Mr = xi[init:fin].size
            val, counts = cp.unique(y[init:fin], return_counts=True)
            maxr = counts.max()
            isum = isum + (maxr / Mr) * maxr
        return isum / n

    def index_from_scheme(self, scheme, x_sorted):
        split_points = []
        for p in scheme:
            split_points.append(cp.where(x_sorted > p)[0][0])
        return split_points

    def check_categorical(self, X, y):
        categorical = []
        ny2 = 2 * cp.unique(y).size
        for j in range(X.shape[1]):
            xj = X.iloc[:, j].dropna().values
            if cp.unique(xj).size < ny2:
                categorical.append(j)
        return categorical


class CategoricalParamException(Exception):
    # Raise if wrong type of parameter
    pass


class NotEnoughPoints(Exception):
    # Raise if a feature must be categorical, not continuous
    pass
