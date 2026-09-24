"""Fit-on-train, apply-elsewhere preprocessing. Every fitted object (imputer,
scaler) is fit exactly once, on train, and reused via apply_* everywhere
else — the original notebook's bug was refitting the imputer on the unseen
set instead of reusing the one fit on train."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config


def fit_median_imputer(df: pd.DataFrame, columns: list[str]) -> SimpleImputer:
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df[columns])
    return imputer


def apply_imputer(df: pd.DataFrame, columns: list[str], imputer: SimpleImputer) -> pd.DataFrame:
    df = df.copy()
    df[columns] = imputer.transform(df[columns])
    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str] = config.OUTLIER_COLUMNS,
    ratio_threshold: float = config.OUTLIER_MAX_TO_P99_RATIO,
    percentile: float = config.OUTLIER_PERCENTILE,
) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        p99 = df[col].quantile(percentile)
        if p99 > 0 and df[col].max() / p99 > ratio_threshold:
            df = df[df[col].isna() | (df[col] < p99)]
    return df


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
