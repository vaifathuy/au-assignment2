from numcompute.io import load_csv
from pathlib import Path
import numpy as np
import pytest


def test_load_csv_with_not_exist_file():
    filename = "nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(filename)


def test_load_csv_with_empty_content():
    file_name = str(Path.cwd()) + "/tests/test_files/test-empty.csv"
    with pytest.raises(FileExistsError):
        load_csv(file_name, delimiter=",")


def test_load_csv_with_data_containing_delimiter():
    file_name = str(Path.cwd()) + "/tests/test_files/test-products-10.csv"
    data = load_csv(file_name, delimiter=",", dtype=(str, float))
    assert type(data) is np.ndarray and data.shape == (10, 13)


def test_load_csv_with_missing_values():
    file_name = str(Path.cwd()) + \
        "/tests/test_files/test-products-10-missing.csv"
    data = load_csv(file_name, delimiter=",", dtype=(str, float))
    assert type(data) is np.ndarray and data.shape == (10, 13)


def test_load_csv_with_semicolon_delimiter():
    file_name = str(Path.cwd()) + \
        "/tests/test_files/test-products-semicolon-delimiter-10.csv"
    data = load_csv(file_name, delimiter=";", dtype=(str, float))
    assert type(data) is np.ndarray and data.shape == (10, 13)


def test_load_csv_with_header():
    file_name = str(Path.cwd()) + \
        "/tests/test_files/test-products-10.csv"
    data = load_csv(file_name, delimiter=",",
                    dtype=(str, float), skip_header=False)
    assert type(data) is np.ndarray and data.shape == (11, 13)
