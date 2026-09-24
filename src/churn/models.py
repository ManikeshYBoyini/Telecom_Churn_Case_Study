"""Train/tune each model behind the same GridSearchCV/RandomizedSearchCV
interface. XGBoost uses scale_pos_weight (computed from the actual class
ratio) instead of the original notebook's class_weight kwarg, which
XGBClassifier's sklearn API does not support."""
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from . import config


def train_logistic_regression(X_train, y_train) -> GridSearchCV:
    folds = StratifiedKFold(n_splits=config.LR_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid=config.LR_PARAM_GRID,
        scoring="roc_auc",
        cv=folds,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def train_random_forest(X_train, y_train) -> GridSearchCV:
    folds = StratifiedKFold(n_splits=config.RF_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    estimator = RandomForestClassifier(
        random_state=config.RANDOM_SEED, n_jobs=-1, class_weight=config.RF_CLASS_WEIGHT
    )
    search = GridSearchCV(
        estimator, param_grid=config.RF_PARAM_GRID, scoring="roc_auc", cv=folds, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search


def train_xgboost(X_train, y_train) -> RandomizedSearchCV:
    folds = StratifiedKFold(n_splits=config.XGB_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = negative / positive if positive else 1.0

    estimator = xgb.XGBClassifier(
        n_jobs=-1,
        objective="binary:logistic",
        random_state=config.RANDOM_SEED,
        scale_pos_weight=scale_pos_weight,
    )
    search = RandomizedSearchCV(
        estimator,
        param_distributions=config.XGB_PARAM_GRID,
        n_iter=config.XGB_N_ITER,
        scoring="roc_auc",
        cv=folds,
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
    )
    search.fit(X_train, y_train)
    return search
