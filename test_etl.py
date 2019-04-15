"""
Title:       test_etl.py
Description: Test code for checking the ETL output.
Author:      Kunyu He
"""

import etl
import pytest
import pandas as pd

FILE_NAME = "./clean_data/credit-clean.csv"


#----------------------------------------------------------------------------#
def check_file(full_path, extension):
    """
    Check whether a file is in the right format, contains at least some
    information, and is readable.
    Inputs:
        - full_path (string): path to the file
        - extension (string): e.g. ".txt"
    Returns:
        (None) make assertions if any condition fails
    """
    if not full_path.endswith(extension):
        raise AssertionError()
    if not os.path.getsize(full_path) > 0:
        raise AssertionError()
    if not os.path.isfile(full_path) and os.access(full_path, os.R_OK):
        raise AssertionError()


#----------------------------------------------------------------------------#
def test_file_accessible():
    """
    Test whether the data file is accessible for further anaylysis.
    """
    data_types = etl.go()
    check_file(FILE_NAME, ".csv")


def test_no_missing():
    """
    Test whether there are no missing value in the clean data.
    """
    data_types = etl.go()
    data = pd.read_csv(FILE_NAME, dtype=data_types)
    if not all(data.isnull().sum() == 0):
        raise AssertionError()
