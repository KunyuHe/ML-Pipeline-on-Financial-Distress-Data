"""
Title:       test_featureEngineering.py
Description: Test code for checking the feature engineering output.
Author:      Kunyu He, CAPP'20
"""

import pytest
import os
import subprocess
import featureEngineering
import numpy as np

from test_etl import check_file
from viz import read_clean_data
from train import load_features


TEST_DIR = "./processed_data/"
TEST_ACCESS = os.listdir(TEST_DIR)
TEST_DISCRETIZE = [('age', 5), ('age', 10)]
TEST_ONE_HOT = [['zipcode'], ['age']]
TEST_SCALING = [load_features()[0]]


#----------------------------------------------------------------------------#
@pytest.mark.parametrize("file_name", TEST_ACCESS)
def test_file_accessible(file_name):
    """
    Test whether the output data files are accessible for further anaylysis.
    """
    subprocess.call('python3 etl.py', shell=True)

    check_file(TEST_DIR + file_name)


@pytest.mark.parametrize("var,bins", TEST_DISCRETIZE)
def test_discretize(var, bins):
    """
    Test whether the function for discretizing a continuous variable into a
    specific number of bins works properly.
    """
    data = read_clean_data()
    discretized = featureEngineering.discretize(data[var], bins)

    if not len(discretized.value_counts()) == bins:
        raise AssertionError()


@pytest.mark.parametrize("cat_vars", TEST_ONE_HOT)
def test_one_hot(cat_vars):
    """
    Test whether the function to create binary/dummy variables from
    categorical variables works properly.
    """
    data = read_clean_data()

    col_counts = data.shape[1]
    drop_counts = len(cat_vars)
    add_counts = 0
    for var in cat_vars:
        add_counts += len(data[var].value_counts())

    processed_data = featureEngineering.one_hot(data, cat_vars)
    if col_counts - drop_counts + add_counts != processed_data.shape[1]:
        raise AssertionError()


@pytest.mark.parametrize("matrix", TEST_SCALING)
def test_scaling(matrix):
    """
    Test whether the function to fit and transform training features with
    standard scaler works properly.
    """
    if not all(abs(matrix.mean(axis=0)) < np.finfo(np.float32).eps):
        raise AssertionError()
