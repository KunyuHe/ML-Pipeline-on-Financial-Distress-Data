"""
Title:       train.py
Description: Collection of functions to train the model.
Author:      Kunyu He, CAPP'20
"""

import numpy as np
import sys

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


INPUT_DIR = "./processed_data/"

MODEL_NAMES = ["KNN", "Decision Tree", "Random Forest"]
MODELS = [KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]
METRICS = ["roc_auc", "accuracy"]

GRID_SEARCH_PARAMS = {"KNN": {
                              'n_neighbors': list(range(10, 55, 5)),
                              'p': [1, 2]
                              },
                      "Decision Tree": {
                            'criterion': ["entropy", "gini"],
                            'min_samples_split': list(np.arange(0.01, 0.06, 0.01)),
                            'max_depth': list(range(1, 11)),
                            'max_features': list(range(4, 17, 2))
                            },
                      "Random Forest": {
                            'min_samples_split': list(np.arange(0.01, 0.06, 0.01)),
                            'max_depth': list(range(4, 11)),
                            'max_features': list(range(4, 17, 2))
                            }
                      }

DEFAULT_ARGS = {"KNN": {'n_jobs': -1},
                "Decision Tree": {'random_state': 123},
                "Random Forest": {'n_estimators': 300, 'random_state': 123,
                                  'oob_score': True}}

#----------------------------------------------------------------------------#
def ask():
    """
    """
    model_index = int(input(("Up till now we support:\n"
                             "1. KNeighborsClassifier\n"
                             "2. DecisionTreeClassifier\n"
                             "3. RandomForestClassifier.\n"
                             "We use default DecisionTreeClassifier as the benchmark.\n"
                             "Please input a classifier index (1, 2, or 3):\n")))

    metric_index = int(input(("Up till now we use two metrics to evaluate the"
                              " fitted classifiers on the test set.\n"
                              "1. Area Under the Receiver Operating Characteristic Curve (ROC AUC)\n"
                              "2. Accuracy.\n"
                              "Please input a metrics index (1, 2, or 3):\n")))

    folds = int(input(("We use grid search and cross-validation on the training"
                       " set to do a exhaustive search over specified parameter"
                       " values for a classifier to find the best set.\n"
                       "Notice that KNN testing really slow and Random Forest"
                       " use a large number of estimators so please use at"
                       " most 5 for these two.\n"
                       "Please input the number of folds for cross-validation, for example, 3, 5, or 10:\n"
                       )))
    return model_index, metric_index, folds


def load_features():
    """
    Load pre-processed feature matrices.
    """
    Xs = np.load(INPUT_DIR + 'X.npz')
    ys = np.load(INPUT_DIR + 'y.npz')
    X_train, X_test = Xs['train'], Xs['test']
    y_train, y_test = ys['train'], ys['test']

    return X_train, X_test, y_train, y_test


def tune(model, parameters, X_train, y_train, metric="roc_auc", n_folds=10,
         default_args={}):
    """
    Use grid search and cross validation to find the best set of hyper-
    parameters.
    """
    classifier = model(**default_args)
    grid = GridSearchCV(classifier, param_grid=parameters, scoring=metric,
                        n_jobs=-1, cv=n_folds, iid=True, verbose=5)

    grid.fit(X_train, y_train)

    return model(**grid.best_params_, **default_args), grid


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    model_index, metric_index, folds = ask()
    X_train, X_test, y_train, y_test = load_features()

    model = MODELS[model_index - 1]
    model_name = MODEL_NAMES[model_index - 1]
    metric_name = METRICS[metric_index - 1]

    parameters = GRID_SEARCH_PARAMS[model_name]
    args = [parameters, X_train, y_train]
    op_args = {'metric': metric_name, 'n_folds': folds}
    default_args = DEFAULT_ARGS[model_name]

    benchmark_classifier = DecisionTreeClassifier(**default_args)
    benchmark_scores = cross_val_score(benchmark_classifier, X_train, y_train,
                                       scoring=metric_name, cv=folds)
    print("{} of the benchmark default decision tree model is {}.".\
          format(metric_name.title(), round(benchmark_scores.mean(), 4)))

    best_classifier, grid = tune(model, *args, **op_args, default_args=default_args)
    print("Found the best set of parameters for {} Classifier: {}".\
          format(model_name, grid.best_params_))

    best_classifier.fit(X_train, y_train)
    best_scores = cross_val_score(best_classifier, X_train, y_train,
                                  scoring=metric_name, cv=folds)
    diff = round(best_scores.mean() - benchmark_scores.mean(), 4)
    print("{} of the tuned {} is {}, {} {} than the benchmark.".\
          format(metric_name, model_name, round(best_scores.mean(), 4), diff,
                 ['higher', 'lower'][int(diff <= 0)]))

    _ = input("Press any key to exit.")
