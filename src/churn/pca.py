"""Fit PCA on train only; transform everything else with that fitted
object. explained_variance_curve is a standalone diagnostic (the scree
plot) and does not feed into the fitted model used elsewhere."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

from . import config


def fit_pca(X_train: pd.DataFrame, n_components: int = config.N_PCA_COMPONENTS) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=n_components)
    pca.fit(X_train)
    return pca


def transform_pca(X: pd.DataFrame, pca: IncrementalPCA) -> np.ndarray:
    return pca.transform(X)


def explained_variance_curve(X_train: pd.DataFrame, random_state: int = config.RANDOM_SEED) -> np.ndarray:
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_train)
    return np.cumsum(pca_full.explained_variance_ratio_)
