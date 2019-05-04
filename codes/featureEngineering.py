"""
Summary:     A collections of functions to generate features.

Description: 
Author:      Kunyu He, CAPP'20
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from viz import read_data


pd.set_option('mode.chained_assignment', None)


INPUT_DIR = "../data/"
OUTPUT_DIR = "../processed_data/"

DATA_FILE = "credit-data.csv"
DATA_TYPES = "data_types.json"

MAX_OUTLIERS = {'MonthlyIncome': 1,
                'DebtRatio': 1}

TO_BINARIES = {'NumberOfTime30-59DaysPastDueNotWorse': 0,
               'NumberOfTime60-89DaysPastDueNotWorse': 0,
               'NumberOfTimes90DaysLate': 0}

TO_COMBINE = {'PastDue': ['NumberOfTimes90DaysLate',
                          'NumberOfTime60-89DaysPastDueNotWorse',
                          'NumberOfTime30-59DaysPastDueNotWorse']}
TO_ORDINAL = {'PastDue': True}

TO_DESCRETIZE = {'age': 5}
RIGHT_INCLUSIVE = {'age': True}

TO_ONE_HOT = ['zipcode']

TARGET = 'SeriousDlqin2yrs'
TO_DROP = ['PersonID', 'SeriousDlqin2yrs']

SPLIT_PARAMS = {'test_size': 0.25, 'random_state': 123}

SCALERS = [StandardScaler, MinMaxScaler]


#----------------------------------------------------------------------------#
def drop_max_outliers(data, drop_vars):
    """
    Takes a data set and a dictionary mapping names of variables whose large
    extremes need to be dropped to number of outliers to drop, drops those
    values.

    Inputs:
        - data (DataFrame): data matrix to modify.
        - drop_vars {string: int}: mapping names of variables to number of
            large outliers to drop.

    Returns:
        (DataFrame) the modified data set.

    """
    for col_name, n in drop_vars.items():
        data.drop(data.nlargest(n, col_name, keep='all').index,
                  axis=0, inplace=True)

    return data


def to_binary(data, to_bin_vars):
    """
    Takes a data set and a dict of variables that needed to be trasformed to
    binaries, performs the transformations and returns the modified data set.

    Inputs:
        - data (DataFrame): data matrix to modify.
        - to_bin_vars ({string: float}): mapping names of variables that
            needed to be trasformed to binaries to cut points.

    Returns:
        (DataFrame): the modified data matrix.

    """
    for var, cut_point in to_bin_vars.items():
        data[var] = np.where(data[var] > cut_point, 1, 0)

    return data


def combine_binaries(data, combine_vars, ordered):
    """
    Takes a data set and a list of lists of binary variables to combine,
    combines them into a ordinal or categorical variable.

    Inputs:
        - data (DataFrame): data matrix to modify.
        - combine_vars ({lists of strings: string}): mapping names of
            variables that needed to be combined to their resulting names.
        - ordered ({string: bool}): mapping names of the resulting columns to
            whether they should be ordered or not.

    Returns:
        (DataFrame): the modified data matrix.

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
        - data (DataFrame): data matrix to modify.
        - cat_vars (list of strings): categorical variables to apply one-hot
            encoding.

    Returns:
        (DataFrame): the modified data matrix.

    """
    for var in cat_vars:
        dummies = pd.get_dummies(data[var], prefix=var)
        data.drop(var, axis=1, inplace=True)
        data = pd.concat([data, dummies], axis=1)

    return data


def split(data, to_drop, test=True, split_params=SPLIT_PARAMS):
    """
    Drop rows with missing labels, split the features and targert, and split data
    into training and test sets if asked to.

    Inputs:
        - data (DataFrame): the data matrix.
        - to_drop (list of strings): columns to drop.
        - test (bool): whether to split the data into training and testing set.
        - split_params (dict): mapping arguments for train test split to values.

    Outputs:
        (X_train, X_test, y_train, y_test)

    """
    data.dropna(axis=0, subset=[TARGET], inplace=True)
    y = data[TARGET]
    data.drop(to_drop, axis=1, inplace=True)
    X = data

    if test:
        return train_test_split(X, y, **split_params)

    return X, None, y, None


def impute(X_train, X_test, data_types, ask=True):
    """
    Take the data matrix, and impute the missing features with a customized
    column feature, then set the data types as preferred from the data
    dictionary.

    Inputs:
        - X_train (arry): training features.
        - X_test (array): testing features.
        - data_types (dict): mapping column name
        - ask (bool): whether to ask for imputer index from the user.

    Returns:
        (DataFrame, DataFrame) the modified training features and test
            features.

    """
    if ask:
        imputer_index = int(input(("Up till now we support:\n"
                                   "\t1. Imputing with column mean\n"
                                   "\t2. Imputing with column median\n"
                                   "Please input the index (1 or 2) of your"
                                   " imputation method:\n")))
    else:
        imputer_index = 1
    data_types = {col: data_type for col, data_type in data_types.items()
                  if col in X_train.columns}

    if imputer_index == 1:
        X_train = X_train.fillna(X_train.mean()).astype(data_types)
        X_test = X_test.fillna(X_test.mean()).astype(data_types)
    else:
        X_train = X_train.fillna(X_train.median()).astype(data_types)
        X_test = X_test.fillna(X_test.median()).astype(data_types)
    print("Finished imputing missing values with feature {}.\n".\
          format(["means", "medians"][imputer_index - 1]))

    return X_train, X_test


def discretize(X_train, X_test, cont_vars, right_inclusive):
    """
    Discretizes a continuous variable into a specific number of bins.

    Inputs:
        - X_train (arry): training features.
        - X_test (array): testing features.
        - cont_vars ({string: int}): mapping column names to number of bins to
            cut the columns in.
        - right_inclusive ({string: bool}): mapping column names to whether to
            cut the series into right inclusive of not.

    Returns:
        (DataFrame) the modified data set.

    """
    for col_name, n in cont_vars.items():
        X_train[col_name] = pd.cut(X_train[col_name], n,
                                   right=right_inclusive[col_name]).cat.codes
        X_test[col_name] = pd.cut(X_test[col_name], n,
                                  right=right_inclusive[col_name]).cat.codes

    return X_train, X_test


def scale(X_train, X_test, ask=True, scale_test=False):
    """
    Asks user for the scaler to use, or uses default standard scaler. Fits it
    on the training data and scales the test data with it if test data is
    provided.
    
    Inputs:
        - X_train (arry): training features.
        - X_test (array): testing features.
        - ask (bool): whether to ask for scaler index from the user.

    Returns:
        (array, array): train and test data after scaling.

    """
    if ask:
        scaler_index = int(input(("\nUp till now we support:\n"
                                  "\t1. StandardScaler\n"
                                  "\t2. MinMaxScaler\n"
                                  "Please input a scaler index (1 or 2):\n")))
    else:
        scaler_index = 1

    scaler = SCALERS[scaler_index - 1]()
    X_train = scaler.fit_transform(X_train.values.astype(float))
    if scale_test:
        X_test = scaler.transform(X_test.values.astype(float))
    print("Finished extracting the target and scaling the features.\n")

    return X_train, X_test


def save_data(X_train, X_test, y_train, y_test, dir_path=OUTPUT_DIR):
    """
    Saves traning and testing data as numpy arrays in the output directory.

    Inputs:
        - X_train (array): training features.
        - X_test (array): testing features.
        - y_train (array): training target.
        - y_test (array): testing target.
        - dir_path (string): relative path to the output directory.

    Returns:
        None

    """
    if "processed_data" not in os.listdir("../"):
        os.mkdir("processed_data")

    np.savez(dir_path + 'X.npz', train=X_train, test=X_test)
    np.savez(dir_path + 'y.npz', train=y_train.values.astype(float),
                                 test=y_test.values.astype(float))

    print(("Saved the resulting NumPy matrices to directory {}. Features are"
           " in 'X.npz' and target is in 'y.npz'.").format(dir_path))


def process():
    """
    Reads the data set, drops rows with large extremes, converts some
    variables into binaries, combines some binaries into categoricals or
    ordinals, and applies one-hot encoding on categoricals. Then splits the
    data set in to training and test, impute missing values separately, and
    descretizes some continuous variables into ordinals. Then save the data as
    NumPy arrays to the output directory.

    Inputs:
        - data (DataFrame): data matrix to modify.

    Returns:
        (DataFrame): the processed data matrix.

    """
    # load data
    data, data_types = read_data(DATA_TYPES, DATA_FILE, dir_path=INPUT_DIR)
    print("Finished reading cleaned data.\n")

    # drop rows with large extremes
    data = drop_max_outliers(data, MAX_OUTLIERS)
    print("Finished dropping extreme large values:")
    for col_name, n in MAX_OUTLIERS.items():
        print("\tDropped {} observations with extreme large values on '{}'.".\
              format(n, col_name))

    # convert some variables into binaries
    data = to_binary(data, TO_BINARIES)
    print("\nFinished transforming the following variables: {}.\n".\
          format(list(TO_BINARIES.keys())))

    # combine some binaries into categoricals or ordinals
    data = combine_binaries(data, TO_COMBINE, TO_ORDINAL)
    print("Finished combining binaries:")
    for col_name, to_combine in TO_COMBINE.items():
        print("\tCombined {} into a {} variable '{}'.".format(to_combine, \
            ["categorical", "ordinal"][int(TO_ORDINAL[col_name])], col_name))

    # apply one-hot encoding on categoricals
    data = one_hot(data, TO_ONE_HOT)
    print("\nFinished one-hot encoding the following categorical variables: {}\n".\
          format(TO_ONE_HOT))

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = split(data, TO_DROP)

    # do imputation to fill in the missing values
    X_train, X_test = impute(X_train, X_test, data_types)

    # discretize some continuous features into ordinals
    X_train, X_test = discretize(X_train, X_test, TO_DESCRETIZE, RIGHT_INCLUSIVE)
    print("Finished discretizing some continuous variables:")
    for col_name, n in TO_DESCRETIZE.items():
        print("\tDiscretized '{}' into {} bins.".format(col_name, n))

    # scale training and test data
    X_train, X_test = scale(X_train, X_test, scale_test=True)

    save_data(X_train, X_test, y_train, y_test)


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    process()
