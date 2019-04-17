"""
Title:       etl.py
Description: A collections of functions for ETL.
Author:      Kunyu He, CAPP'20
"""

import pandas as pd
import os
import sys
import json

INPUT_DIR = "./data/"
OUTPUT_DIR = "./clean_data/"


#----------------------------------------------------------------------------#
def translate_data_type(data_type):
    """
    Translate a data type from the data dictionary to a domain in pandas
    DataFrame
    """
    if data_type in ["integer", "Y/N"]:
        return "int"
    elif data_type in ["percentage", "real"]:
        return "float"
    return "object"


def load_data():
    """
    Load data into a pandas DataFrame, extract data types from the data
    dictionary, fill missing values and modify data types.
    """
    data_dict = pd.read_excel(INPUT_DIR + "Data Dictionary.xls", header=1)
    types = data_dict.Type.apply(translate_data_type)
    data_types = dict(zip(data_dict['Variable Name'], types))

    data = pd.read_csv(INPUT_DIR + "credit-data.csv")
    data.fillna(data.median(), inplace=True)
    
    return data.astype(data_types), data_types


def go():
    """
    Read data, apply changes and write it into a new csv file. Also write data
    dictionary to a json file.
    """
    data, data_types = load_data()
    if "clean_data" not in os.listdir():
        os.mkdir("clean_data")

    data.to_csv(OUTPUT_DIR + "credit-clean.csv", index=False)
    with open(OUTPUT_DIR + "data_type.json", 'w') as file:
        json.dump(data_types, file)
    

#----------------------------------------------------------------------------#
if __name__ == "__main__":
    go()

    print(("ETL process finished. Data wrote to 'credit-clean.csv'"
           " and 'data_type.json' under directory 'clean_data'."
           " Missing values are filled in with medians of the columns"))
