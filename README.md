# Machine Learning Pipeline on Financial Distress Data

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9012ccbbd6e64987807a44a0f828e33b)](https://app.codacy.com/app/kunyuhe/ML-Pipeline-on-Financial-Distress-Data?utm_source=github.com&utm_medium=referral&utm_content=KunyuHe/ML-Pipeline-on-Financial-Distress-Data&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data.svg?branch=master)](https://travis-ci.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data)
[![Maintainability](https://api.codeclimate.com/v1/badges/d9e3f244250a2f44e012/maintainability)](https://codeclimate.com/github/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/maintainability)
[![codecov](https://codecov.io/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/branch/master/graph/badge.svg)](https://codecov.io/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/master?filepath=.%2FEDA%2FData%20Exploration.ipynb)

## 0. To Reproduce My Results

Folk and clone the repository to your local machine.

### 0.1 Windows

```
$ chmod u+x run.sh
$ run.sh
```

### 0.2 Unix/Linux

```
$ chmod +x script.sh
$ ./run.sh
```

## 1. Introduction

The task is to build a pipeline that predicts whether an individual will experience financial distress (experience 90 days past due delinquency or worse) in the next two years, based on income, household and credit history of that person.

The pipeline has six components:

1.  Read Data
2.  Explore Data
3.  Pre-Process Data
4.  Generate Features/Predictors
5.  Build Classifier
6.  Evaluate Classifier

The pipeline currently supports **three classification algorithms** (KNN, Decision Tree, and Random Forest), and use grid search and cross validation to find the best model for the future in terms of either of the **two evaluation metrics** (Accuracy or Area Under the Receiver Operating Characteristic Curve).

Details would be covered in the following sections.

## 2. Get Data

*  Output Directory: `./data/`

Data is manually downloaded from the UChicago canvas website as given. It is stored in the `./data/` directory as `credit-data.csv` and `Data Dictionary.xls`.

## 3. Read Data

*  Input Directory: `./data/`
*  Output Directory: `./clean_data/`
*  Script: [etl.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/etl.py)
*  Test Script: [test_etl.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/test_etl.py)

As data comes as CSV, I used `Pandas` to read it into Python. Meanwhile, data types are given in the data dictionary, but not in a form that `Pandas` would understand. So in module `etl.py` I include two functions, one that translates data types in the data dictionary to a `Pandas` data type ("int", "float", or "object") and stores them into a `.json` file, and another that read the data, **fill missing values with column median**.

## 4. Explore Data

*  Input Directory: `./clean_data/`
*  Notebook: [Data Exploration.ipynb](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/master?filepath=.%2FEDA%2FData%20Exploration.ipynb)
*  Script: [viz.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/viz.py)
*  Test Script: [test_viz.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/test_viz.py)

**Try the interactive Jupyter Notebook supported by binder if you click on the badge above**!

For the EDA process, I generated
barplots for the categorical variables, and used histograms to show the distributions of variables. I also drawed a correlation triangle to show the correlations between them, and did a panel of box plots to find outliers.

