"""
Summary:     A collections of functions for ETL.

Description: This module is used for loading the data, loading and translating
             data types from the data dictionary, and saving them to the
             output directory "../clean_data/".
Author:      Kunyu He, CAPP'20
"""

import os
import json
import pandas as pd


INPUT_DIR = "../data/"
OUTPUT_DIR = "../data/"

DATA_DICTIONARY = "Data Dictionary.xls"
HEADER_ROW = 1


#----------------------------------------------------------------------------#
def translate_data_type(data_type):
    """
    Summary: Translate a data type from the data dictionary to a domain in
    pandas DataFrame.

    Inputs:
        - data_type (string): data type in the data dictionary.

    Returns:
        (string) data type for pandas DataFrame.

    """
    if data_type in ["integer", "Y/N"]:
        return "int"

    if data_type in ["percentage", "real"]:
        return "float"

    return "object"


def save_dictionary(data_dictonary, header_row):
    """
    Load data into a pandas DataFrame, extract data types from the data
    dictionary, fill missing values and modify data types.

    Inputs:
        - data_dictonary (string): name of the data dictionary.
        - header_row (int): row of the dictionary headers.

    Returns:
        None

    """
    data_dict = pd.read_excel(INPUT_DIR + data_dictonary, header=header_row)
    types = data_dict.Type.apply(translate_data_type)
    data_types = dict(zip(data_dict['Variable Name'], types))

    with open(OUTPUT_DIR + "data_types.json", 'w') as file:
        json.dump(data_types, file)

    print(("ETL process finished. Data dictionary wrote to 'data_types.json'"
           " under the directory {}.".format(OUTPUT_DIR)))


def perform_etl():
    """
    Perform the ETL process.

    Inputs:
        None

    Returns:
        None

    """
    save_dictionary(DATA_DICTIONARY, HEADER_ROW)


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    perform_etl()
