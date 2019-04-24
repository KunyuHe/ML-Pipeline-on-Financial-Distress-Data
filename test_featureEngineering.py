"""
Title:       test_featureEngineering.py
Description: Test code for checking the feature engineering output.
Author:      Kunyu He, CAPP'20
"""

import os
import pytest
import featureEngineering

from test_etl import check_file
from viz import read_clean_data


INPUT_DIR = "./clean_data/"
TEST_DIR = "./processed_data/"
TEST_ACCESS = os.listdir(TEST_DIR)
TEST_DISCRETIZE = [('age', 5), ('NumberOfDependents', 3)]
TEST_ONE_HOT = [['zipcode'], ['age'], ['zipcode', 'age']]
TEST_TO_BINARY = [{'NumberOfTime30-59DaysPastDueNotWorse': 0,
                   'NumberOfTime60-89DaysPastDueNotWorse': 0,
                   'NumberOfTimes90DaysLate': 0},
                  {'age': 18},
                  {'NumberOfDependents': 0}]


#----------------------------------------------------------------------------#
@pytest.mark.parametrize("file_name", TEST_ACCESS)
def test_file_accessible(file_name):
    """
    Test whether the output data files are accessible for further anaylysis.
    """
    check_file(TEST_DIR + file_name)


@pytest.mark.parametrize("var,bins", TEST_DISCRETIZE)
def test_discretize(var, bins):
    """
    Test whether the function for discretizing a continuous variable into a
    specific number of bins works properly.
    """
    data = read_clean_data("data_types.json", "credit-clean.csv",
                           dir_path=INPUT_DIR)
    discretized = featureEngineering.discretize(data[var], bins)

    if not len(discretized.value_counts()) == bins:
        raise AssertionError()


@pytest.mark.parametrize("cat_vars", TEST_ONE_HOT)
def test_one_hot(cat_vars):
    """
    Test whether the function to create binary/dummy variables from
    categorical variables works properly.
    """
    data = read_clean_data("data_types.json", "credit-clean.csv",
                           dir_path=INPUT_DIR)

    col_counts = data.shape[1]
    drop_counts = len(cat_vars)
    add_counts = 0
    for var in cat_vars:
        add_counts += len(data[var].value_counts())

    processed_data = featureEngineering.one_hot(data, cat_vars)
    if col_counts - drop_counts + add_counts != processed_data.shape[1]:
        raise AssertionError()


@pytest.mark.parametrize("to_bin_vars", TEST_TO_BINARY)
def test_to_binary(to_bin_vars):
    """
    Test whether the function to trasform variables to binaries
    works properly.
    """
    data = read_clean_data("data_types.json", "credit-clean.csv",
                           dir_path=INPUT_DIR)
    data = featureEngineering.to_binary(data, to_bin_vars)

    for var in to_bin_vars:
        if len(data[var].value_counts()) > 2:
            raise AssertionError()
