# Machine Learning Pipeline on Financial Distress Data

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9012ccbbd6e64987807a44a0f828e33b)](https://app.codacy.com/app/kunyuhe/ML-Pipeline-on-Financial-Distress-Data?utm_source=github.com&utm_medium=referral&utm_content=KunyuHe/ML-Pipeline-on-Financial-Distress-Data&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data.svg?branch=master)](https://travis-ci.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data)
[![Maintainability](https://api.codeclimate.com/v1/badges/d9e3f244250a2f44e012/maintainability)](https://codeclimate.com/github/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/maintainability)
[![codecov](https://codecov.io/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/branch/master/graph/badge.svg)](https://codecov.io/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data)
[![Documentation Status](https://readthedocs.org/projects/pydocstyle/badge/?version=stable)](http://www.pydocstyle.org/en/stable/?badge=stable)
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
3.  Generate Features/Predictors
4.  Build Classifier
5.  Evaluate Classifier

The pipeline currently supports **seven classification algorithms**:

*   KNN Classifier
*   Logistic Regression Classifier
*   Decision Tree Classifier
*   Linear SVC
*   Bagging
*   Boosting
*   Random Forest Classifier

Upon decision on type of the model, it uses grid search and cross validation to find the best set of hyperparameters to build the best model for the future, in terms of either of the **five evaluation metrics**:

*   Accuracy
*   Precision
*   Recall
*   F-1 Score
*   AUC ROC

Users can run the program either with their specific configurations, or run all possible combinations at once. **Please pay attention to the console prompts and make your choice.**

During the training process, **three/four plots that help the evaluation of a specific model would prompt for `3 seconds` and close automatically**, including:

*   Distribution of the Predicted Probablities
*   Precision, Recall Curve and Percent of Polpulation
*   Receiver Operating Characteristic Curve
*   Feature Importance (Top 5) Bar Plots (if applicable)

**Please don't close them manually**, or the program won't build.

Details would be covered in the following sections.

## 2. Get Data

*   Output Directory: `./data/`

Data is manually downloaded from the UChicago canvas website as given. It is stored in the `./data/` directory as `credit-data.csv` and `Data Dictionary.xls`.

There are 41016 observations and 12 original features in the data set. A list of the features and corresponding descriptions are available in the [data dictionary](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/data/Data%20Dictionary.xls?raw=true).

## 3. Read Data

*   Input Directory: `./data/`
*   Output Directory: `./clean_data/`
*   Code Script: [etl.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/etl.py)
*   Test Script: [test_etl.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/test_etl.py)

As data comes as CSV, I used `Pandas` to read it into Python. Meanwhile, data types are given in the data dictionary, but not in a form that `Pandas` would understand. So in module `etl.py` I include two functions, one that translates data types in the data dictionary to a `Pandas` data type ("int", "float", or "object") and stores them into a `.json` file, and another that read the data, **fill missing values with column median**.

## 4. Explore Data

*   Input Directory: `./clean_data/`
*   Output Directory: `./images/`
*   Notebook: [Data Exploration.ipynb](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/master?filepath=.%2FEDA%2FData%20Exploration.ipynb)
*   Code Script: [viz.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/viz.py)
*   Test Script: [test_viz.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/test_viz.py)

**Try the interactive Jupyter Notebook supported by binder if you click on the badge above**!

For the EDA process, I generated barplots for the categorical variables, and used histograms to show the distributions of variables. I also drawed a correlation triangle to show the correlations between them, and did a panel of box plots to find the outliers. The images are saved in the `./images/` directory.

## 5. Feature Engineering

*   Input Directory: `./clean_data/`
*   Output Directory: `./processed_data/`
*   Code Script: [featureEngineering.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/featureEngineering.py)
*   Test Script: [test_featureEngineering.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/test_featureEngineering.py)

I've filled in the missing values in the `Read Data` phase. In this part, I got rid of the outliers and generate some new features.

![](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/EDA/images/figure-3.png)

As observed above, there are two large exterme values in `MonthlyIncome` and `DebtRatio`, hence I drop those observations from the data set.

![](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/EDA/images/figure-2.png)

Also, as number of times borrower go past due in the last 2 years are strongly correlated, I transformed them into binaries and combine those dummies to generate a new feature `PastDue`. `PastDue` is ordinal and indicates the degree of the individual's past financial distress. While 0 indicates he has never been past due for over 30 days in the past two years, 3 indicates he at least once experienced 90 days past due delinquency or worse.

For the algorithms this pipeline currently supports, there's no need to transform all categorical variables into dummies, but only those without inherent ordering. So I transformed the categorical variable `zipcode` into a set of dummies. I also applied a function that can discretize continuous variables on variable `age` and turned it into a five-level ordinal variable. I used all of the resulting independent variables as my features and extracted the target `SeriousDlqin2yrs`.

I used the method of cross-validation to evaluate models in the pipeline. As feature scaling is crucial for KNN, and would make the training process of Decision Tree and Random Forest faster, **user can choose to use either `StandardScaler` or `MinMaxScaler` to normalize the features matrix**. Both features matrix and target vector is stored as `.npz` file for future use.

## 6. Build and Evaluate Classifier

*   Input Directory: `./processed_data/`
*   Code Script: [train.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/train.py)
*   Test Script: *In Progess*

For the training and evaluation part, I built the benchmark with a default scikit-learn `DecisionTreeClassifier`. User can build a semi-customized classifier when they choose the algorithm to use (KNN, Decision Tree, and Random Forest), the evaluation metrics for hyper-parameter tuning and cross-validation testing, and the number of folds of cross-validation during grid search and testing.

For each of the classifiers that the pipline supports, I pre-set a few default paramters (`random_state`, `n_estimators`...) and designed a grid of parameters for model tuning. After users choose a model, a evaluation metrics, and number of cross-validation folds, **the program would build the benchmark, instantiate a classifier and exhaustively search over the specified grid to find the best set of paramter values in terms of cross-validation scoring for it, and fit it with the best set of parameters. It would further print the average cross-validation score and compare it with that of the benchmark model.**

## 7. Results

The result depends largely on user's choices. For example, in one run the user choose to use the `StandardScaler`, train a `DecisionTreeClassifier`, with `Accuracy` as the evaluation metrics, and do a `5-fold` cross-validation. **The average cross-validation accuracy is 0.7034 for the benchmark classifier, 0.8739 for the tuned decision tree, which is higher by 0.1705.** It takes about 3 minutes for the program to finish running.
