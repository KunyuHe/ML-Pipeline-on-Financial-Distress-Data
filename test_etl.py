"""
Title:       test_etl.py
Description: Test code for checking the ETL output.
Author:      Kunyu He, CAPP'20
"""

import os
import pytest
import etl
import pandas as pd


TEST_DIR = "./clean_data/"

TEST_ACCESS = os.listdir(TEST_DIR)
TEST_NO_MISSING = ["credit-clean.csv"]


#----------------------------------------------------------------------------#
def check_file(full_path):
    """
    Check whether a file contains at least some information and is readable.
    Inputs:
        - full_path (string): path to the file
    Returns:
        (None) make assertions if any condition fails
    """
    if not os.path.getsize(full_path) > 0:
        raise AssertionError()
    if not os.path.isfile(full_path) and os.access(full_path, os.R_OK):
        raise AssertionError()


#----------------------------------------------------------------------------#
@pytest.mark.parametrize("file_name", TEST_ACCESS)
def test_file_accessible(file_name):
    """
    Test whether the data file is accessible for further anaylysis.
    """
    etl.go()
    check_file(TEST_DIR + file_name)


@pytest.mark.parametrize("file_name", TEST_NO_MISSING)
def test_no_missing(file_name):
    """
    Test whether there are no missing value in the clean data.
    """
    etl.go()
    data = pd.read_csv(TEST_DIR + file_name)
    if not all(data.isnull().sum() == 0):
        raise AssertionError()
