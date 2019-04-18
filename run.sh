#!/bin/bash

python etl.py

start https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/master?filepath=.%2FEDA%2FData%20Exploration.ipynb

python featureEngineering.py

echo Up till now we support 1. KNeighborsClassifier, 2. DecisionTreeClassifier, 3. RandomForestClassifier.
echo We use default DecisionTreeClassifier as the benchmark for testing.
echo Please input a classifier index:
read classifier_index

echo Up till now we use two metrics to evaluate the fitted classifiers on the test set.
ehco 1. Area Under the Receiver Operating Characteristic Curve (ROC AUC) 2. Accuracy.
echo Please input a metrics index:
read metrics_index

read -p "Press any key to exit" x

echo We use grid search and cross-validation on the training set to do a exhaustive search over specified parameter values for a classifier to find the best set.
echo Please input the number of folds for cross-validation, for example, 3, 5, or 10:
echo Notice that KNN testing really slow and Random Forest use a large number of estimators so please use at most 5 for these two.
read num_folds

#python train.py $classifier_index, $metrics_index, $num_folds

read -p "Press any key to exit" x
