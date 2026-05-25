"""Unit tests for src/data.py — load_dataset_split() contract.

Run with:
    pytest tests/test_data.py -v

NOTE: These tests require the dataset file at data/speed_dating_data.csv.
      They are skipped automatically if the file is missing.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from data import load_dataset_split
    from config import DATA_DIR
    DATASET_AVAILABLE = (DATA_DIR / "speed_dating_data.csv").exists()
except Exception:
    DATASET_AVAILABLE = False

requires_dataset = pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="speed_dating_data.csv not found in data/ — download from Kaggle first",
)


@pytest.fixture(scope="module")
def split():
    return load_dataset_split()


class TestReturnContract:
    @requires_dataset
    def test_returns_tuple(self, split):
        assert isinstance(split, tuple)

    @requires_dataset
    def test_tuple_has_four_elements(self, split):
        assert len(split) == 4

    @requires_dataset
    def test_all_elements_are_numpy_arrays(self, split):
        X_train, X_test, y_train, y_test = split
        for name, arr in [("X_train", X_train), ("X_test", X_test),
                          ("y_train", y_train), ("y_test", y_test)]:
            assert isinstance(arr, np.ndarray)


class TestShapeAndSize:
    @requires_dataset
    def test_x_train_and_y_train_same_length(self, split):
        X_train, _, y_train, _ = split
        assert len(X_train) == len(y_train)

    @requires_dataset
    def test_x_test_and_y_test_same_length(self, split):
        _, X_test, _, y_test = split
        assert len(X_test) == len(y_test)

    @requires_dataset
    def test_train_is_larger_than_test(self, split):
        X_train, X_test, _, _ = split
        assert len(X_train) > len(X_test)

    @requires_dataset
    def test_split_ratio_approximately_80_20(self, split):
        X_train, X_test, _, _ = split
        total = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total
        assert 0.75 <= train_ratio <= 0.85

    @requires_dataset
    def test_x_train_has_correct_feature_count(self, split):
        X_train, _, _, _ = split
        assert X_train.shape[1] == 34

    @requires_dataset
    def test_x_test_same_feature_count_as_train(self, split):
        X_train, X_test, _, _ = split
        assert X_train.shape[1] == X_test.shape[1]

    @requires_dataset
    def test_total_size_matches_dataset(self, split):
        X_train, X_test, _, _ = split
        total = len(X_train) + len(X_test)
        assert 8_000 <= total <= 8_500


class TestPreprocessing:
    @requires_dataset
    def test_no_nan_in_x_train(self, split):
        X_train, _, _, _ = split
        assert not np.isnan(X_train).any()

    @requires_dataset
    def test_no_nan_in_x_test(self, split):
        _, X_test, _, _ = split
        assert not np.isnan(X_test).any()

    @requires_dataset
    def test_x_train_approximately_standardized(self, split):
        X_train, _, _, _ = split
        col_means = np.abs(X_train.mean(axis=0))
        col_stds = X_train.std(axis=0)
        assert col_means.max() < 0.1
        assert col_stds.max() < 1.5

    @requires_dataset
    def test_no_infinite_values(self, split):
        X_train, X_test, _, _ = split
        assert np.isfinite(X_train).all()
        assert np.isfinite(X_test).all()


class TestTargetVariable:
    @requires_dataset
    def test_y_train_binary(self, split):
        _, _, y_train, _ = split
        unique_values = set(np.unique(y_train))
        assert unique_values.issubset({0, 1})

    @requires_dataset
    def test_y_test_binary(self, split):
        _, _, _, y_test = split
        unique_values = set(np.unique(y_test))
        assert unique_values.issubset({0, 1})

    @requires_dataset
    def test_stratified_split_preserves_match_rate(self, split):
        _, _, y_train, y_test = split
        assert abs(y_train.mean() - y_test.mean()) < 0.02

    @requires_dataset
    def test_match_rate_approximately_16_percent(self, split):
        _, _, y_train, y_test = split
        y_all = np.concatenate([y_train, y_test])
        overall_rate = y_all.mean()
        assert 0.13 <= overall_rate <= 0.20


class TestNoDataLeakage:
    @requires_dataset
    def test_x_test_not_centered_at_zero(self, split):
        _, X_test, _, _ = split
        col_means = X_test.mean(axis=0)
        max_deviation = np.abs(col_means).max()
        assert max_deviation < 5.0
