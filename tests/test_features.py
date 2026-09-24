import pandas as pd
from churn import features


def _sample_df():
    return pd.DataFrame({
        "aon": [365, 730, 1095, 1460],          # 12, 24, 36, 48 months
        "arpu_6": [100, 200, 300, 400],
        "arpu_7": [100, 200, 300, 400],
        "arpu_8": [100, 200, 300, 400],
    })


def test_avg_customer_spend_combines_aon_months_and_mean_arpu():
    df = _sample_df()
    result, _ = features.engineer_customer_value(df)
    # row 0: aon_months=12, mean_arpu=100 -> avg_customer_Spend=1200
    assert result["avg_customer_Spend"].iloc[0] == 1200.0


def test_original_columns_are_dropped():
    df = _sample_df()
    result, _ = features.engineer_customer_value(df)
    for col in ["aon", "arpu_6", "arpu_7", "arpu_8"]:
        assert col not in result.columns


def test_thresholds_computed_on_first_call_are_reused_on_second_call():
    train_df = _sample_df()
    train_result, thresholds = features.engineer_customer_value(train_df)

    # a wildly different "unseen" frame must bucket using TRAIN's thresholds
    unseen_df = pd.DataFrame({
        "aon": [3650], "arpu_6": [10000], "arpu_7": [10000], "arpu_8": [10000],
    })
    unseen_result, reused_thresholds = features.engineer_customer_value(unseen_df, thresholds=thresholds)

    assert reused_thresholds == thresholds
    assert unseen_result["customer_value"].iloc[0] == 2  # HVC — far above train's HVC threshold


def test_customer_value_encoding_is_0_1_2_for_lvc_mvc_hvc():
    df = _sample_df()
    result, _ = features.engineer_customer_value(df)
    # spends [1200, 4800, 10800, 19200] against hvc=10500.0..., lvc=3360.0:
    # row0 LVC(0), row1 MVC(1), row2 HVC(2), row3 HVC(2)
    assert list(result["customer_value"]) == [0, 1, 2, 2]
