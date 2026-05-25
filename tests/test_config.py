"""Unit tests for src/config.py — MODELS registry contract.

Run with:
    pytest tests/test_config.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

import config
from config import MODELS, MODELS_DIR, MODEL_METRICS_FILE, STREAMLIT_HOST, STREAMLIT_PORT


class TestModelsRegistry:
    def test_models_is_dict(self):
        assert isinstance(MODELS, dict)

    def test_models_is_not_empty(self):
        assert len(MODELS) > 0

    def test_each_entry_has_path_key(self):
        for key, entry in MODELS.items():
            assert "path" in entry

    def test_path_values_are_path_objects(self):
        for key, entry in MODELS.items():
            assert isinstance(entry["path"], Path)

    def test_path_values_are_absolute(self):
        for key, entry in MODELS.items():
            assert entry["path"].is_absolute()

    def test_path_values_under_models_dir(self):
        for key, entry in MODELS.items():
            assert str(entry["path"]).startswith(str(MODELS_DIR))

    def test_path_values_have_valid_extension(self):
        valid_extensions = {".pkl", ".pickle", ".joblib"}
        for key, entry in MODELS.items():
            suffix = entry["path"].suffix.lower()
            assert suffix in valid_extensions

    def test_name_key_is_string_if_present(self):
        for key, entry in MODELS.items():
            if "name" in entry:
                assert isinstance(entry["name"], str)

    def test_description_key_is_string_if_present(self):
        for key, entry in MODELS.items():
            if "description" in entry:
                assert isinstance(entry["description"], str)

    def test_model_keys_are_strings(self):
        for key in MODELS:
            assert isinstance(key, str)

    def test_random_forest_registered(self):
        assert "random_forest" in MODELS

    def test_logistic_regression_registered(self):
        assert "logistic_regression" in MODELS


class TestDirectoryConfig:
    def test_models_dir_is_path(self):
        assert isinstance(MODELS_DIR, Path)

    def test_model_metrics_file_is_path(self):
        assert isinstance(MODEL_METRICS_FILE, Path)

    def test_model_metrics_file_has_csv_extension(self):
        assert MODEL_METRICS_FILE.suffix == ".csv"

    def test_results_dir_created_on_import(self):
        assert MODEL_METRICS_FILE.parent.exists()


class TestStreamlitConfig:
    def test_streamlit_host_is_string(self):
        assert isinstance(STREAMLIT_HOST, str)

    def test_streamlit_port_is_int(self):
        assert isinstance(STREAMLIT_PORT, int)

    def test_streamlit_port_in_valid_range(self):
        assert 1024 <= STREAMLIT_PORT <= 65535
