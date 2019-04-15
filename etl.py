"""
Title:       etl.py
Description: A collections of functions for ETL.
Author:      Kunyu He, CAPP'20
"""

import pandas as pd
import os
import sys


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
    data_dict = pd.read_excel("Data Dictionary.xls", header=1)
    types = data_dict.Type.apply(translate_data_type)
    data_types = dict(zip(data_dict['Variable Name'], types))

    data = pd.read_csv("credit-data.csv")
    data.fillna(data.median(), inplace=True)
    
    return data.astype(data_types), data_types


def go():
    """
    Read data, apply changes and write it into a new csv file nam 
    """
    os.chdir("./data")
    data, data_types = load_data()

    os.chdir("..")
    if "clean_data" not in os.listdir():
        os.mkdir("clean_data")
    os.chdir("./clean_data")

    data.to_csv("credit-clean.csv", index=False)
    return data_types


#----------------------------------------------------------------------------#
if __name__ == "__main__":
    data_types = go()
    print(("ETL process finished. Data wrote to 'credit-clean.csv'"
           " under directory 'clean_data'."))
