import pandas as pd
from churn import data


def _sample_df():
    # 10 rows so percentages are exact. mostly_missing has 2 distinct
    # non-null values (pandas' nunique() ignores NaN, so a single repeated
    # non-null value here would make it look "constant" too — it must have
    # >=2 distinct values to isolate the missing-values check from the
    # constant-columns check).
    return pd.DataFrame({
        "id": list(range(1, 11)),
        "mostly_missing": [1.0, 2.0, 1.0, None, None, None, None, None, None, None],  # 70% missing, nunique=2
        "constant": [7] * 10,                                                          # nunique == 1
        "signup_date": [f"1/{i}/2020" for i in range(1, 11)],
        "good_numeric": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "churn_probability": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    })


def test_get_high_missing_columns_flags_over_threshold():
    df = _sample_df()
    assert data.get_high_missing_columns(df, threshold_pct=60.0) == ["mostly_missing"]


def test_get_constant_columns_flags_single_unique_value():
    df = _sample_df()
    assert data.get_constant_columns(df) == ["constant"]


def test_get_date_columns_flags_object_dtype():
    df = _sample_df()
    assert data.get_date_columns(df) == ["signup_date"]


def test_compute_drop_columns_combines_all_three_plus_id():
    df = _sample_df()
    drop_cols = set(data.compute_drop_columns(df, threshold_pct=60.0))
    assert drop_cols == {"mostly_missing", "constant", "signup_date", "id"}


def test_apply_column_drops_is_safe_on_a_frame_missing_one_flagged_column():
    train_df = _sample_df()
    drop_cols = data.compute_drop_columns(train_df, threshold_pct=60.0)

    # a "test-like" frame that never had the constant column to begin with
    test_df = train_df.drop(columns=["constant", "churn_probability"])
    result = data.apply_column_drops(test_df, drop_cols)

    assert list(result.columns) == ["good_numeric"]


def test_load_train_reads_csv_from_given_path(tmp_path):
    csv_path = tmp_path / "mini_train.csv"
    _sample_df().to_csv(csv_path, index=False)
    loaded = data.load_train(path=csv_path)
    assert loaded.shape == (10, 6)
