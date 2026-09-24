import numpy as np
import pandas as pd
from churn import pca as pca_mod


def _sample_df(n_rows=20, n_cols=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"c{i}" for i in range(n_cols)])


def test_fit_pca_returns_object_that_transforms_to_requested_components():
    train_df = _sample_df()
    pca_model = pca_mod.fit_pca(train_df, n_components=3)
    transformed = pca_mod.transform_pca(train_df, pca_model)
    assert transformed.shape == (20, 3)


def test_transform_never_refits_on_the_new_data():
    train_df = _sample_df(seed=0)
    pca_model = pca_mod.fit_pca(train_df, n_components=2)
    mean_after_fit = pca_model.mean_.copy()

    other_df = _sample_df(seed=99) * 1000 + 500  # wildly different scale/location
    pca_mod.transform_pca(other_df, pca_model)

    # transform_pca must not have mutated the fitted object's learned mean —
    # if it had refit on other_df, this would differ substantially.
    assert np.allclose(pca_model.mean_, mean_after_fit)


def test_explained_variance_curve_is_nondecreasing_and_ends_near_one():
    train_df = _sample_df(n_cols=4)
    curve = pca_mod.explained_variance_curve(train_df)
    assert np.all(np.diff(curve) >= -1e-9)
    assert curve[-1] > 0.99
