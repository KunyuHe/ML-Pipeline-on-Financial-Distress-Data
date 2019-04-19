"""
Title:       featureEngineering.py
Description: A collections of functions to generate features.
Author:      Kunyu He, CAPP'20
"""

import pandas as pd
import numpy as np
import os

from viz import read_clean_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


INPUT_DIR = "./clean_data/"
OUTPUT_DIR = "./processed_data/"

TO_DESCRETIZE = ['age']
TO_ONE_HOT = ['zipcode']
TARGET = 'SeriousDlqin2yrs'
TO_DROP = ['PersonID', 'SeriousDlqin2yrs']
SCALERS = [StandardScaler, MinMaxScaler]


#----------------------------------------------------------------------------#
def discretize(ds, bins, right_inclusive=True, digit=3):
    """
    Discretize a continuous variable into a specific number of bins.
    """
    return pd.cut(ds, bins, right=right_inclusive, precision=digit).cat.codes


def one_hot(data, cat_vars):
    """
    Take a data set and a list of categorical variables, create binary/dummy
    variables from them and insert them back to the data set.
    """
    for var in cat_vars:
        dummies = pd.get_dummies(data[var], prefix=var)
        data.drop(var, axis=1, inplace=True)
        data = pd.concat([data, dummies], axis=1)

    return data


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    data = read_clean_data()
    print("Finished reading cleaned data.")

    for var in TO_DESCRETIZE:
        data[var] = discretize(data[var], 5, digit=0)
    print("Finished discretizing the following continuous variables: {}".\
          format(TO_DESCRETIZE))

    processed_data = one_hot(data, TO_ONE_HOT)
    print("Finished one-hot encoding the following categorical variables: {}".\
          format(TO_ONE_HOT))

    y_train = processed_data[TARGET].values.astype(float)
    for var in TO_DROP:
        processed_data.drop(var, axis=1, inplace=True)
    X_train = processed_data.values.astype(float)

    scaler_index = int(input(("Up till now we support:\n"
                              "1. StandardScaler\n"
                              "2. MinMaxScaler\n"
                              "Please input a scaler index (1 or 2):\n")))
    X_train = SCALERS[scaler_index - 1]().fit_transform(X_train)
    print("Finished extracting the target and scaling the features.")

    if "processed_data" not in os.listdir():
        os.mkdir("processed_data")
    np.savez(OUTPUT_DIR + 'X.npz', train=X_train, test=None)
    np.savez(OUTPUT_DIR + 'y.npz', train=y_train, test=None)
    print(("Saved the resulting NumPy matrices to directory {}. Features are"
           " in 'X.npz' and target is in 'y.npz'.").format(OUTPUT_DIR))
