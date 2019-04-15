"""
Title:       etl.py
Description: A collections of functions for ETL.
Author:      Kunyu He, CAPP'20
"""

import pandas as pd
import os
import sys


#============================================================================#
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


def load_data(fn):
    """
    Load data into a pandas DataFrame, extract data types from the data
    dictionary, fill missing values and modify data types.
    """
    data_dict = pd.read_excel("Data Dictionary.xls", header=1)
    types = data_dict.Type.apply(translate_data_type)
    data_types = dict(zip(data_dict['Variable Name'], types))

    data = pd.read_csv("credit-data.csv")
    if fn == "median":
        data.fillna(data.median(), inplace=True)
    data.fillna(data.mean(), inplace=True)
    
    return data.astype(data_types)


def go(fn):
    """
    Read data, apply changes and write it into a new csv file nam 
    """
    data = load_data(fn)

    os.chdir("..")
    if "clean_data" not in os.listdir():
        os.mkdir("clean_data")
    os.chdir("./clean_data")

    data.to_csv("credit-clean.csv", index=False)


#============================================================================#
if __name__ == "__main__":
    os.chdir("./data")

    fn = sys.argv[1]
    if fn not in ["mean", "median"]:
        print("usage: python3 {} mean (or median)".format(sys.argv[0]))
        sys.exit(1)

    go(fn)
    print(("ETL process finished. Data wrote to 'credit-clean.csv'"
           " under directory 'clean_data'."))
