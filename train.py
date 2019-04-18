"""
Title:       train.py
Description: Collection of functions to train the model.
Author:      Kunyu He, CAPP'20
"""

import numpy as np
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV


INPUT_DIR = "./processed_data/"
MODEL_NAMES = ["KNN", "Decision Tree", "Random Forest"]
METRICS = ["ROC AUC", "Accuracy"]
GRID_SEARCH_PARAMS = {"KNN": {'n_neighbors': list(range(10, 55, 5)),
                              'p': [1, 2]
                              },
                      "Decision Tree": {'criterion': ["entropy", "gini"],
                            'min_samples_split': list(np.arange(0.01, 0.11, 0.01)),
                            'max_depth': list(range(1, 11)),
                            'max_features': list(range(4, 17, 2))
                            },
                      "Random Forest": {'min_samples_split': list(np.arange(0.01, 0.06, 0.01)),
                            'max_depth': list(range(4, 11)),
                            'max_features': list(range(4, 17, 2))
                            }
                      }


#----------------------------------------------------------------------------#
def load_features():
    """
    Load pre-processed feature matrices.
    """
    Xs = np.load(INPUT_DIR + 'X.npz')
    ys = np.load(INPUT_DIR + 'y.npz')
    X_train, X_test = Xs['train'], Xs['test']
    y_train, y_test = ys['train'], ys['test']

    return X_train, X_test, y_train, y_test


def evaluate(classifier, X_test, y_test, metric="ROC AUC"):
    """
    Evaluate the fitted classifier on the test set and calculate the
    evaluation metrics.
    """
    y_pred = classifier.predict(X_test)

    if metric == "Accuracy":
        return accuracy_score(y_test, y_pred)

    fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred)
    return auc(fp_rate, tp_rate)


def build_benchmark(X_train, y_train, X_test, y_test, metric="ROC AUC"):
    """
    """
    benchmark_classifier = DecisionTreeClassifier(random_state=123)
    benchmark_classifier.fit(X_train, y_train)

    return evaluate(benchmark_classifier, X_test, y_test, metric=metric)


def tune(model, parameters, X_train, y_train, metric="ROC AUC", n_folds=5,
         default_args={}):
    """
    Use grid search and cross validation to find the best set of hyper-
    parameters.
    """
    classifier = model(**default_args)
    if metric == "ROC AUC":
        score = "roc_auc"
    else:
        score = "accuracy"

    grid = GridSearchCV(classifier, param_grid=parameters, scoring=score,
                        n_jobs=-1, cv=n_folds, iid=True, verbose=5)
    grid.fit(X_train, y_train)
    
    return model(**grid.best_params_, **default_args), grid


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_features()
    model_index, metric_index, folds = sys.argv[1:]
    model_name = MODEL_NAMES[model_index - 1]
    metric_name = METRICS[metric_index - 1]

    benchmark_score = build_benchmark(X_train, y_train, X_test, y_test,
                                      metric=metric_name)
    print("{} of the benchmark default decision tree model is {}.".\
          format(metric_name, round(benchmark_score, 3)))

    parameters = GRID_SEARCH_PARAMS[model_name]
    args = [parameters, X_train, y_train]
    op_args = {'metric': metric_name, 'n_folds': folds}

    if model_name == "KNN":
        best_classifier, grid = tune(KNeighborsClassifier, *args, **op_args,
                                     default_args={'n_jobs': -1})
    elif model_name == "Decision Tree":
        best_classifier, grid = tune(DecisionTreeClassifier, *args, **op_args,
                                     default_args={'random_state': 123})
    else:
        best_classifier, grid = tune(RandomForestClassifier, *args, **op_args,
                                     default_args={'n_estimators': 1000,
                                                   'random_state': 123,
                                                   'oob_score': True})
    print("Found the best set of parameters for {} Classifier: {}".\
          format(model_name, grid.best_params_))

    best_classifier.fit(X_train, y_train)
    best_score = evaluate(best_classifier, X_test, y_test, metric=metric_name)
    diff = round(best_score - benchmark_score, 3)
    print("{} of the tuned {} is {}, {} {} than the benchmark.".\
          format(metric_name, model_name, round(best_score, 3), diff,
                 ['higher', 'lower'][int(diff <= 0)]))
