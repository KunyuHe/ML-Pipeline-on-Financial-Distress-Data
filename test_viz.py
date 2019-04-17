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
numeric = data[[col for col in data.columns if col not in NON_NUMERIC]]

TEST_BAR = [target.value_counts(), data.zipcode.value_counts()]
TEST_OTHERS = [(numeric, viz.hist_panel),
               (pd.concat([target, numeric], axis=1), viz.corr_triangle)]


#----------------------------------------------------------------------------#
@pytest.mark.parametrize("ds", TEST_BAR)
def test_bar_plot(ds):
    """
    Test whether the bar plotting works fine.
    """
    _, ax = plt.subplots()
    viz.bar_plot(ax, ds)

    if not plt.gcf().number == 1:
        raise AssertionError()

    plt.close('all')


@pytest.mark.parametrize("data, fn", TEST_OTHERS)
def test_others(data, fn):
    """
    Test whether the histogram panel plotting and correlation triangle
    plotting work fine.
    """
    fn(data)

    if not plt.gcf().number == 1:
        raise AssertionError()

    plt.close('all')
