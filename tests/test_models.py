import numpy as np
import pandas as pd
from churn import models


def _sample_Xy(n_rows=60, n_cols=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"c{i}" for i in range(n_cols)])
    # imbalanced-ish target, but with enough of the minority class for 3-fold CV
    y = pd.Series([0] * 40 + [1] * 20)
    return X, y


def test_train_logistic_regression_returns_fitted_search_with_probability_predict():
    X, y = _sample_Xy()
    search = models.train_logistic_regression(X, y)
    assert hasattr(search, "best_estimator_")
    proba = search.best_estimator_.predict_proba(X)
    assert proba.shape == (60, 2)


def test_train_random_forest_returns_fitted_search_with_probability_predict():
    X, y = _sample_Xy()
    search = models.train_random_forest(X, y)
    assert hasattr(search, "best_estimator_")
    proba = search.best_estimator_.predict_proba(X)
    assert proba.shape == (60, 2)


def test_train_xgboost_returns_fitted_search_with_probability_predict():
    X, y = _sample_Xy()
    search = models.train_xgboost(X, y)
    assert hasattr(search, "best_estimator_")
    proba = search.best_estimator_.predict_proba(X)
    assert proba.shape == (60, 2)
