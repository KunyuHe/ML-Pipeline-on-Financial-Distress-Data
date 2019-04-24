"""
Title:       etl.py
Description: A collections of functions for ETL.
Author:      Kunyu He, CAPP'20
"""

import os
import json
import pandas as pd


INPUT_DIR = "./data/"
OUTPUT_DIR = "./clean_data/"


#----------------------------------------------------------------------------#
def translate_data_type(data_type):
    """
    Translate a data type from the data dictionary to a domain in pandas
    DataFrame.

    Inputs:
        - data_type (string): data type in the data dictionary

    Returns:
        (string) data type for pandas DataFrame
    """
    if data_type in ["integer", "Y/N"]:
        return "int"

    if data_type in ["percentage", "real"]:
        return "float"

    return "object"


def load_data(data_dictonary, data_file):
    """
    Load data into a pandas DataFrame, extract data types from the data
    dictionary, fill missing values and modify data types.

    Inputs:
        - data_dictonary (string): name of the data dictionary.
        - data_file (string): name of the data file.

    Returns:
        (DataFrame, dict) clean data set with missing values filled by
            column median, dictionary mapping column names to data types
    """
    data_dict = pd.read_excel(INPUT_DIR + data_dictonary, header=1)
    types = data_dict.Type.apply(translate_data_type)
    data_types = dict(zip(data_dict['Variable Name'], types))

    data = pd.read_csv(INPUT_DIR + data_file)
    data.fillna(data.median(), inplace=True)

    return data.astype(data_types), data_types


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    data, data_types = load_data("Data Dictionary.xls", "credit-data.csv")
    if "clean_data" not in os.listdir():
        os.mkdir("clean_data")

    data.to_csv(OUTPUT_DIR + "credit-clean.csv", index=False)
    with open(OUTPUT_DIR + "data_types.json", 'w') as file:
        json.dump(data_types, file)

    print(("ETL process finished. Data wrote to 'credit-clean.csv'"
           " and 'data_types.json' under directory 'clean_data'."
           " Missing values are filled in with medians of the columns."))
