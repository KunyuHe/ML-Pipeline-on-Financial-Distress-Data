"""
Title:       test_viz.py
Description: Test code for checking functions for visualization.
Author:      Kunyu He, CAPP'20
"""

import pytest
import pandas as pd
import matplotlib.pyplot as plt

import viz


INPUT_DIR = "./clean_data/"
NON_NUMERIC = ["PersonID", "zipcode", "SeriousDlqin2yrs"]

data = viz.read_clean_data("data_types.json", "credit-clean.csv",
                           dir_path=INPUT_DIR)
target = data.SeriousDlqin2yrs
numeric = data[[col for col in data.columns if col not in NON_NUMERIC]]
nvt = pd.concat([target, numeric], axis=1)

TEST_BAR = [(data, 'SeriousDlqin2yrs'), (data, 'zipcode')]
TEST_OTHER = [numeric]


#----------------------------------------------------------------------------#
@pytest.mark.parametrize("data,column", TEST_BAR)
def test_bar_plot(data, column):
    """
    Test whether the bar plotting works fine.
    """
    _, ax = plt.subplots()
    viz.bar_plot(ax, data, column)

    if not plt.gcf().number == 1:
        raise AssertionError()

    plt.close('all')


@pytest.mark.parametrize("data", TEST_OTHER)
def test_hists(data):
    """
    Test whether the histogram panel plotting works fine.
    """
    viz.hist_panel(data)

    if not plt.gcf().number == 1:
        raise AssertionError()

    plt.close('all')


@pytest.mark.parametrize("data", TEST_OTHER)
def test_corr(data):
    """
    Test whether the correlation triangle plotting works fine.
    """
    _, ax = plt.subplots()
    viz.corr_triangle(ax, data)

    if not plt.gcf().number == 1:
        raise AssertionError()

    plt.close('all')
