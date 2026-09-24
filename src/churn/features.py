"""avg_customer_Spend and the HVC/MVC/LVC customer_value bucket, reproduced
from the original notebook's logic. The HVC/LVC quantile thresholds are
computed once (on train, when thresholds=None) and must be passed back in
on every later call so an unseen customer's bucket depends only on what was
learned from training data."""
import pandas as pd

from . import config


def engineer_customer_value(
    df: pd.DataFrame, thresholds: tuple[float, float] | None = None
) -> tuple[pd.DataFrame, tuple[float, float]]:
    df = df.copy()
    aon_months = df["aon"] / 365 * 12
    mean_arpu = (df["arpu_6"] + df["arpu_7"] + df["arpu_8"]) / 3
    df["avg_customer_Spend"] = aon_months * mean_arpu
    df = df.drop(columns=["aon", "arpu_6", "arpu_7", "arpu_8"])

    if thresholds is None:
        hvc = df["avg_customer_Spend"].quantile(config.HVC_QUANTILE)
        lvc = df["avg_customer_Spend"].quantile(config.LVC_QUANTILE)
    else:
        hvc, lvc = thresholds

    df["customer_value"] = df["avg_customer_Spend"].apply(
        lambda x: 2 if x > hvc else (0 if x < lvc else 1)
    )
    return df, (hvc, lvc)
