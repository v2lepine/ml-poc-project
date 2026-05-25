"""Unit tests for src/metrics.py — compute_metrics() contract.

Run with:
    pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from metrics import compute_metrics


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    return y_true, y_pred


@pytest.fixture
def all_wrong_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    return y_true, y_pred


@pytest.fixture
def imbalanced_predictions():
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=200, p=[0.836, 0.164])
    y_pred = y_true.copy()
    flip_idx = rng.choice(np.where(y_true == 0)[0], size=10, replace=False)
    y_pred[flip_idx] = 1
    flip_idx2 = rng.choice(np.where(y_true == 1)[0], size=5, replace=False)
    y_pred[flip_idx2] = 0
    return y_true, y_pred


@pytest.fixture
def always_negative():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1])
    y_pred = np.zeros(9, dtype=int)
    return y_true, y_pred


class TestReturnContract:
    def test_returns_dict(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_dict_is_not_empty(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        assert len(result) > 0

    def test_required_keys_present(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        required_keys = {"f1", "precision", "recall", "accuracy"}
        assert required_keys.issubset(result.keys())

    def test_all_values_are_float(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        for key, value in result.items():
            assert isinstance(value, float)

    def test_all_values_are_convertible_to_float(self, imbalanced_predictions):
        y_true, y_pred = imbalanced_predictions
        result = compute_metrics(y_true, y_pred)
        for key, value in result.items():
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                pytest.fail(f"float({key}={value!r}) raised {exc}")


class TestMetricValues:
    def test_perfect_f1_is_one(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        assert result["f1"] == pytest.approx(1.0)

    def test_perfect_accuracy_is_one(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_perfect_precision_is_one(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        assert result["precision"] == pytest.approx(1.0)

    def test_perfect_recall_is_one(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        assert result["recall"] == pytest.approx(1.0)

    def test_naive_baseline_accuracy(self, always_negative):
        y_true, y_pred = always_negative
        result = compute_metrics(y_true, y_pred)
        expected_accuracy = 5 / 9
        assert result["accuracy"] == pytest.approx(expected_accuracy, abs=1e-6)
        assert result["f1"] == pytest.approx(0.0)

    def test_all_metrics_between_zero_and_one(self, imbalanced_predictions):
        y_true, y_pred = imbalanced_predictions
        result = compute_metrics(y_true, y_pred)
        for key, value in result.items():
            assert 0.0 <= value <= 1.0

    def test_f1_is_harmonic_mean_of_precision_and_recall(self, imbalanced_predictions):
        y_true, y_pred = imbalanced_predictions
        result = compute_metrics(y_true, y_pred)
        p = result["precision"]
        r = result["recall"]
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert result["f1"] == pytest.approx(expected_f1, abs=1e-6)


class TestRobustness:
    def test_no_predicted_positives(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        result = compute_metrics(y_true, y_pred)
        assert result["precision"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_no_true_positives(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = compute_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_single_sample(self):
        result_pos = compute_metrics(np.array([1]), np.array([1]))
        result_neg = compute_metrics(np.array([0]), np.array([0]))
        assert isinstance(result_pos, dict)
        assert isinstance(result_neg, dict)

    def test_large_input(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=10_000)
        y_pred = rng.integers(0, 2, size=10_000)
        result = compute_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_list_input_accepted(self):
        result = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])
        assert isinstance(result, dict)
