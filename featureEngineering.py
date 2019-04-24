"""
Title:       featureEngineering.py
Description: A collections of functions to generate features.
Author:      Kunyu He, CAPP'20
"""

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from viz import read_clean_data


pd.set_option('mode.chained_assignment', None)


INPUT_DIR = "./clean_data/"
OUTPUT_DIR = "./processed_data/"

MAX_OUTLIERS = {'MonthlyIncome': 1,
                'DebtRatio': 1}
TO_BINARIES = {'NumberOfTime30-59DaysPastDueNotWorse': 0,
               'NumberOfTime60-89DaysPastDueNotWorse': 0,
               'NumberOfTimes90DaysLate': 0}
TO_COMBINE = {'PastDue': ['NumberOfTimes90DaysLate',
                          'NumberOfTime60-89DaysPastDueNotWorse',
                          'NumberOfTime30-59DaysPastDueNotWorse']}
TO_ORDINAL = {'PastDue': True}
TO_DESCRETIZE = ['age']
TO_ONE_HOT = ['zipcode']
TARGET = 'SeriousDlqin2yrs'
TO_DROP = ['PersonID', 'SeriousDlqin2yrs']
SCALERS = [StandardScaler, MinMaxScaler]


#----------------------------------------------------------------------------#
def drop_max_outliers(data, drop_vars):
    """
    Takes a data set and a dictionary mapping names of variables whose large
    extremes need to be dropped to number of outliers to drop, drop those
    values.

    Inputs:
        - data (DataFrame): data matrix to modify
        - drop_vars {string: int}: mapping names of variables to number of
            large outliers to drop

    Returns:
        (DataFrame) the modified data set
    """
    for col_name, n in drop_vars.items():
        data.drop(data.nlargest(n, col_name, keep='all').index,
                  axis=0, inplace=True)

    return data


def discretize(ds, bins, right_inclusive=True, digit=3):
    """
    Discretizes a continuous variable into a specific number of bins.

    Inputs:
        - ds (Series): data series to be discretized
        - bins (int): number of bins to cut the series into
        - right_inclusive (bool): cut the series into right inclusive of left
        - digit (int): digit to round while discretizing

    Returns:
        (Series) of integer codes representing the categories
    """
    return pd.cut(ds, bins, right=right_inclusive, precision=digit).cat.codes


def to_binary(data, to_bin_vars):
    """
    Takes a data set and a dict of variables that needed to be trasformed to
    binaries, performs the transformations and returns the modified data set.

    Inputs:
        - data (DataFrame): data matrix to modify
        - to_bin_vars ({string: float}): mapping names of variables that
            needed to be trasformed to binaries to cut points

    Returns:
        (DataFrame): the modified data matrix
    """
    for var, cut_point in to_bin_vars.items():
        data[var] = np.where(data[var] > cut_point, 1, 0)

    return data


def combine_binaries(data, combine_vars, ordered):
    """
    Takes a data set and a list of lists of binary variables to combine,
    combine them into a ordinal or categorical variable.

    Inputs:
        - data (DataFrame): data matrix to modify
        - combine_vars ({lists of strings: string}): mapping names of
            variables that needed to be combined to their resulting names
        - ordered ({string: bool}): mapping names of the resulting columns to
            whether they should be ordered or not

    Returns:
        (DataFrame): the modified data matrix
    """
    for col_name, to_combine in combine_vars.items():
        dummies = data[to_combine]
        dummies['negative'] = np.where((dummies == 1).sum(axis=1) > 0, 0, 1)

        data[col_name] = pd.Categorical(dummies.idxmax(axis=1),
                                        categories=(to_combine + ['negative'])[::-1],
                                        ordered=ordered[col_name]).codes
        data.drop(to_combine, axis=1, inplace=True)

    return data


def one_hot(data, cat_vars):
    """
    Takes a data set and a list of categorical variables, creates binary/dummy
    variables from them, drops the original and inserts the dummies back to the
    data set.

    Inputs:
        - data (DataFrame): data matrix to modify
        - cat_vars (list of strings): categorical variables to apply one-hot
            encoding

    Returns:
        (DataFrame): the modified data matrix
    """
    for var in cat_vars:
        dummies = pd.get_dummies(data[var], prefix=var)
        data.drop(var, axis=1, inplace=True)
        data = pd.concat([data, dummies], axis=1)

    return data


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    # load data
    data = read_clean_data("data_types.json", "credit-clean.csv",
                           dir_path=INPUT_DIR)
    print("Finished reading cleaned data.\n")

    # drop rows with large extremes
    data = drop_max_outliers(data, MAX_OUTLIERS)
    print("Finished dropping extreme large values:")
    for col_name, n in MAX_OUTLIERS.items():
        print("\tDropped {} observations with extreme large values on '{}'".\
              format(n, col_name))

    # convert some variables into binaries
    data = to_binary(data, TO_BINARIES)
    print("\nFinished transforming the following variables: {}\n".\
          format(list(TO_BINARIES.keys())))

    # combine some binaries into categoricals or ordinals
    data = combine_binaries(data, TO_COMBINE, TO_ORDINAL)
    print("Finished combining binaries:")
    for col_name, to_combine in TO_COMBINE.items():
        print("\tCombined {} into a {} variable '{}'".format(to_combine, \
            ["categorical", "ordinal"][int(TO_ORDINAL[col_name])], col_name))

    # descretize some continuous variables into ordinals
    for var in TO_DESCRETIZE:
        data[var] = discretize(data[var], 5, digit=0)
    print("\nFinished discretizing the following continuous variables: {}\n".\
          format(TO_DESCRETIZE))

    # apply one-hot encoding on categoricals
    processed_data = one_hot(data, TO_ONE_HOT)
    print("Finished one-hot encoding the following categorical variables: {}\n".\
          format(TO_ONE_HOT))

    # extract training data
    y_train = processed_data[TARGET].values.astype(float)
    for var in TO_DROP:
        processed_data.drop(var, axis=1, inplace=True)
    X_train = processed_data.values.astype(float)

    # apply scaling
    scaler_index = int(input(("Up till now we support:\n"
                              "\t1. StandardScaler\n"
                              "\t2. MinMaxScaler\n"
                              "Please input a scaler index (1 or 2):\n")))
    X_train = SCALERS[scaler_index - 1]().fit_transform(X_train)
    print("Finished extracting the target and scaling the features.\n")

    # save output numpy arrays
    if "processed_data" not in os.listdir():
        os.mkdir("processed_data")
    np.savez(OUTPUT_DIR + 'X.npz', train=X_train, test=None)
    np.savez(OUTPUT_DIR + 'y.npz', train=y_train, test=None)
    print(("Saved the resulting NumPy matrices to directory {}. Features are"
           " in 'X.npz' and target is in 'y.npz'.").format(OUTPUT_DIR))
