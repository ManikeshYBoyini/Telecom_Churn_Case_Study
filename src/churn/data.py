"""Loading train/test/data-dictionary CSVs, and computing the column-drop
list ONCE from train so it can be applied identically to test/unseen data —
never recomputed from test, which would let test's own missingness pattern
silently change which columns survive."""
import pandas as pd

from . import config


def load_train(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.TRAIN_CSV)


def load_test(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.TEST_CSV)


def load_data_dictionary(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.DATA_DICTIONARY_CSV)


def get_high_missing_columns(df: pd.DataFrame, threshold_pct: float = config.MISSING_THRESHOLD_PCT) -> list[str]:
    missing_pct = 100 * df.isnull().sum() / len(df)
    return list(df.columns[missing_pct > threshold_pct])


def get_constant_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if df[col].nunique() == 1]


def get_date_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])]


def compute_drop_columns(train_df: pd.DataFrame, threshold_pct: float = config.MISSING_THRESHOLD_PCT) -> list[str]:
    drop_cols = set(get_high_missing_columns(train_df, threshold_pct))
    drop_cols |= set(get_constant_columns(train_df))
    drop_cols |= set(get_date_columns(train_df))
    drop_cols.add(config.ID_COLUMN)
    return sorted(drop_cols)


def apply_column_drops(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    present = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=present)
