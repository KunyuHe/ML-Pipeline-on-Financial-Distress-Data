# Machine Learning Pipeline on Financial Distress Data

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9012ccbbd6e64987807a44a0f828e33b)](https://app.codacy.com/app/kunyuhe/ML-Pipeline-on-Financial-Distress-Data?utm_source=github.com&utm_medium=referral&utm_content=KunyuHe/ML-Pipeline-on-Financial-Distress-Data&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data.svg?branch=master)](https://travis-ci.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data)
[![Maintainability](https://api.codeclimate.com/v1/badges/d9e3f244250a2f44e012/maintainability)](https://codeclimate.com/github/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/maintainability)
[![codecov](https://codecov.io/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/branch/master/graph/badge.svg)](https://codecov.io/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/master?filepath=.%2FEDA%2FData%20Exploration.ipynb)

## 0. To Reproduce My Results

Folk and clone the repository to your local machine. Change you working directory to the cloned folder and run one of the following:

### 0.1 Windows

```console
chmod u+x run.sh
run.sh
```

### 0.2 Unix/Linux

```console
chmod +x script.sh
./run.sh
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

*   Output Directory: `./data/`

Data is manually downloaded from the UChicago canvas website as given. It is stored in the `./data/` directory as `credit-data.csv` and `Data Dictionary.xls`.

## 3. Read Data

*   Input Directory: `./data/`
*   Output Directory: `./clean_data/`
*   Code Script: [etl.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/etl.py)
*   Test Script: [test_etl.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/test_etl.py)

As data comes as CSV, I used `Pandas` to read it into Python. Meanwhile, data types are given in the data dictionary, but not in a form that `Pandas` would understand. So in module `etl.py` I include two functions, one that translates data types in the data dictionary to a `Pandas` data type ("int", "float", or "object") and stores them into a `.json` file, and another that read the data, **fill missing values with column median**.

## 4. Explore Data

*   Input Directory: `./clean_data/`
*   Notebook: [Data Exploration.ipynb](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/master?filepath=.%2FEDA%2FData%20Exploration.ipynb)
*   Code Script: [viz.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/viz.py)
*   Test Script: [test_viz.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/test_viz.py)

**Try the interactive Jupyter Notebook supported by binder if you click on the badge above**!

For the EDA process, I generated barplots for the categorical variables, and used histograms to show the distributions of variables. I also drawed a correlation triangle to show the correlations between them, and did a panel of box plots to find the outliers.

## 5. Pre-Process Data

As I've filled in the missing values in the `Read Data` phase, I skipped this part for now.

## 6. Generate Features/Predictors

*   Input Directory: `./clean_data/`
*   Output Directory: `./processed_data/`
*   Code Script: [featureEngineering.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/featureEngineering.py)
*  Test Script: [test_featureEngineering.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/test_featureEngineering.py)

For the algorithms this pipeline currently supports, there's no need to transform all categorical variables into dummies, but only those without inherent ordering. So I transformed the categorical variable `zipcode` into a set of dummies. I also applied a function that can discretize continuous variables on variable `age` and turned it into a five-level ordinal variable. I used all of the resulting independent variables as my features and extracted the target `SeriousDlqin2yrs`.

I used the method of cross-validation to evaluate models in the pipeline. As feature scaling is crucial for KNN, and would make the training process of Decision Tree and Random Forest faster, **user can choose to use either StandardScaler or MinMaxScaler to normalize the features matrix**. Both features matrix and target vector is stored as `.npz` file for future use.

## 7. Build and Evaluate Classifier
