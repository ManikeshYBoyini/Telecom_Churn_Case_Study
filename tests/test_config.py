from churn import config


def test_paths_are_absolute_and_under_project_root():
    assert config.PROJECT_ROOT.is_absolute()
    assert config.DATA_RAW_DIR == config.PROJECT_ROOT / "data" / "raw"
    assert config.TRAIN_CSV == config.DATA_RAW_DIR / "train.csv"


def test_outlier_columns_cover_all_three_tracked_months():
    for month in ("6", "7", "8"):
        assert any(col.endswith(f"_{month}") for col in config.OUTLIER_COLUMNS)


def test_random_seed_matches_original_notebook():
    assert config.RANDOM_SEED == 42


def test_missing_threshold_matches_original_notebook():
    assert config.MISSING_THRESHOLD_PCT == 60.0
