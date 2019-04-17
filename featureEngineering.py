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
from sklearn.preprocessing import StandardScaler


INPUT_DIR = "./clean_data/"
OUTPUT_DIR = "./processed_data/"

TO_DESCRETIZE = ['age']
TO_ONE_HOT = ['zipcode']
TARGET = 'SeriousDlqin2yrs'
TO_DROP = ['PersonID', 'SeriousDlqin2yrs']


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


def split(processed_data, test_size=0.25, random_state=123):
    """
    Split the processed data (after discretizing and one-hot encoding) into
    training and testing set and wait for scaling.
    """
    y = processed_data[TARGET].values.astype(float)
    for var in TO_DROP:
        processed_data.drop(var, axis=1, inplace=True)
    X = processed_data.values.astype(float)

    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state)


def feature_scaling(X_train, X_test, scaler=StandardScaler()):
    """
    Fit and transform training features with the desired scaler and use the
    trained scaler to transform the test features.
    """
    sc_X = scaler

    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train, X_test


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

    X_train, X_test, y_train, y_test = split(processed_data)
    X_train, X_test = feature_scaling(X_train, X_test)
    print(("Finished extracting the target and scaling the features, and"
           " split them into training and testing sets."))

    if "processed_data" not in os.listdir():
        os.mkdir("processed_data")
    np.savez(OUTPUT_DIR + 'X.npz', train=X_train, test=X_test)
    np.savez(OUTPUT_DIR + 'y.npz', train=y_train, test=y_test)
    print(("Saved the resulting NumPy matrices to directory {}. Features are"
           " in 'X.npz' and target is in 'y.npz'.").format(OUTPUT_DIR))
