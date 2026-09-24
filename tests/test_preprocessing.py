import pandas as pd
from churn import preprocessing


def test_impute_uses_the_fitted_median_not_the_new_frames_median():
    train_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})  # median of [1,2,3] = 2.0
    imputer = preprocessing.fit_median_imputer(train_df, ["x"])

    other_df = pd.DataFrame({"x": [100.0, 200.0, None]})  # median of [100,200] = 150.0
    result = preprocessing.apply_imputer(other_df, ["x"], imputer)

    assert result["x"].iloc[2] == 2.0  # train's median, not other_df's own median


def test_remove_outliers_drops_rows_only_when_ratio_exceeds_threshold():
    # 99 normal values around 10, one extreme value of 1000 -> max/p99 ratio is huge
    df = pd.DataFrame({"total_og_mou_6": [10.0] * 99 + [1000.0]})
    result = preprocessing.remove_outliers(
        df, columns=["total_og_mou_6"], ratio_threshold=6.0, percentile=0.99
    )
    assert len(result) < len(df)
    assert 1000.0 not in result["total_og_mou_6"].values


def test_remove_outliers_keeps_rows_when_ratio_is_within_threshold():
    df = pd.DataFrame({"total_og_mou_6": list(range(1, 101))})  # gentle spread, ratio < 6
    result = preprocessing.remove_outliers(
        df, columns=["total_og_mou_6"], ratio_threshold=6.0, percentile=0.99
    )
    assert len(result) == len(df)


def test_remove_outliers_preserves_nan_rows_instead_of_dropping_them():
    # 99 normal values, one true outlier (ratio > 6 threshold), and one NaN in the
    # same column. NaN < p99 evaluates to False, so a naive filter would drop the
    # NaN row too even though it isn't an outlier -- it just hasn't been imputed yet.
    df = pd.DataFrame({"total_og_mou_6": [10.0] * 99 + [1000.0] + [None]})
    result = preprocessing.remove_outliers(
        df, columns=["total_og_mou_6"], ratio_threshold=6.0, percentile=0.99
    )

    assert 1000.0 not in result["total_og_mou_6"].values  # true outlier removed
    assert result["total_og_mou_6"].isna().sum() == 1  # NaN row survives
    assert len(result) == len(df) - 1  # only the outlier row was dropped


def test_scaler_uses_the_fitted_mean_not_the_new_frames_mean():
    train_df = pd.DataFrame({"x": [0.0, 10.0]})  # mean 5.0, std 5.0
    scaler = preprocessing.fit_scaler(train_df)

    # 20.0 differs from the train mean, so a buggy refit-on-other_df (which would
    # always yield 0.0 for a single-row frame, since sklearn clips zero variance
    # scale to 1) is distinguishable from the correct fit-once-apply-elsewhere result.
    other_df = pd.DataFrame({"x": [20.0]})
    result = preprocessing.apply_scaler(other_df, scaler)

    # (20.0 - train_mean=5.0) / train_std=5.0 -> 3.0, using train's fitted stats
    assert result["x"].iloc[0] == 3.0
