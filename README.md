# ML-Pipeline-on-Financial-Distress-Data

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9012ccbbd6e64987807a44a0f828e33b)](https://app.codacy.com/app/kunyuhe/ML-Pipeline-on-Financial-Distress-Data?utm_source=github.com&utm_medium=referral&utm_content=KunyuHe/ML-Pipeline-on-Financial-Distress-Data&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data.svg?branch=master)](https://travis-ci.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data)
[![Maintainability](https://api.codeclimate.com/v1/badges/d9e3f244250a2f44e012/maintainability)](https://codeclimate.com/github/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/maintainability)
[![codecov](https://codecov.io/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/branch/master/graph/badge.svg)](https://codecov.io/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/master?filepath=.%2FEDA%2FData%20Exploration.ipynb)

## Introduction

The task is to build a pipeline that predicts whether an individual will experience financial distress (experience 90 days past due delinquency or worse) in the next two years, based on income, household and credit history of that person.

The pipeline has six components:

1. Read Data
2. Explore Data
3. Pre-Process Data
4. Generate Features/Predictors
5. Build Classifier
6. Evaluate Classifier

The pipeline currently supports **three classification algorithms** (KNN, Decision Tree, and Random Forest), and use grid search and cross validation to find the best model for the future in terms of either of the **two evaluation metrics** (Accuracy or Area Under the Receiver Operating Characteristic Curve).

Details would be covered in the following sections.

## Read Data
