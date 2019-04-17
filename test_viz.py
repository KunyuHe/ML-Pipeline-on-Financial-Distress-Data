"""
Title:       test_viz.py
Description: Test code for checking functions for visualization.
Author:      Kunyu He
"""

import etl
import json
import pandas as pd
import pytest
import viz
import matplotlib.pyplot as plt

INPUT_DIR = "./clean_data/"
NON_NUMERIC = ["PersonID", "zipcode", "SeriousDlqin2yrs"]

etl.go()
with open(INPUT_DIR + "data_type.json") as file:
    data_types = json.load(file)
data = pd.read_csv(INPUT_DIR + "credit-clean.csv", dtype=data_types)
target = data.SeriousDlqin2yrs

TEST_BAR = [target.value_counts(), data.zipcode.value_counts()]
TEST_HIST = data[[col for col in data.columns if col not in NON_NUMERIC]]
TEST_CORR = pd.concat([target, TEST_HIST], axis=1)


#----------------------------------------------------------------------------#
@pytest.mark.parametrize("ds", TEST_BAR)
def test_bar_plot(ds):
    """
    Test whether the bar plotting works fine.
    """
    fig, ax = plt.subplots()
    viz.bar_plot(ax, ds)

    if not plt.gcf().number == 1:
        raise AssertionError()
    plt.pause(3)
    plt.close()


def test_hist_panels():
    """
    Test whether the histogram panel plotting works fine.
    """
    viz.hist_panel(TEST_HIST)

    if not plt.gcf().number == 1:
        raise AssertionError()
    plt.pause(3)
    plt.close()


def test_corr_plot():
    """
    Test whether the correlation triangle plotting works fine.
    """
    viz.corr_triangle(TEST_CORR)

    if not plt.gcf().number == 1:
        raise AssertionError()
    plt.pause(3)
    plt.close()
