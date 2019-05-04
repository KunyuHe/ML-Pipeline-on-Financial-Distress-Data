"""
Title:       train.py
Description: Collection of functions to train the model.
Author:      Kunyu He, CAPP'20
"""

import numpy as np
import itertools
import warnings

from matplotlib.font_manager import FontProperties
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from viz import plot_predicted_scores, plot_precision_recall

warnings.filterwarnings("ignore")

INPUT_DIR = "../processed_data/"
OUTPUT_DIR = "../log/"

MODEL_NAMES = ["KNN", "Logistic Regression", "Decision Tree", "Linear SVM",
               "Random Forest"]
MODELS = [KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier,
          LinearSVC, RandomForestClassifier]

METRICS_NAMES = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC ROC Score"]
METRICS = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]

THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEED = 123

GRID_SEARCH_PARAMS = {"KNN": {
                              'n_neighbors': list(range(50, 110, 20)),
                              'weights': ["uniform", "distance"],
                              'metric': ["euclidean", "manhattan", "minkowski"]
                              },

                      "Logistic Regression": {
                                              'penalty': ['l1', 'l2'],
                                              'C': [0.001, 0.01, 0.1, 1, 10, 100],
                                              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                                              },

                      "Decision Tree": {
                            'criterion': ["entropy", "gini"],
                            'min_samples_split': list(np.arange(0.02, 0.05, 0.01)),
                            'max_depth': list(range(4, 11)),
                            'max_features': list(range(4, 15, 2))
                            },

                      "Linear SVM": {
                                     'penalty': ['l1', 'l2'],
                                     'C': [0.001, 0.01, 0.1, 1, 10]
                                     },

                      "Random Forest": {
                            'min_samples_split': list(np.arange(0.01, 0.06, 0.01)),
                            'max_depth': list(range(4, 11)),
                            'max_features': list(range(4, 15, 2))
                            }
                      }

DEFAULT_ARGS = {"KNN": {'n_jobs': -1},
                "Logistic Regression": {'random_state': SEED},
                "Decision Tree": {'random_state': SEED},
                "Linear SVM": {'random_state': SEED},
                "Random Forest": {'n_estimators': 300, 'random_state': SEED,
                                  'oob_score': True}}


#----------------------------------------------------------------------------#
def ask():
    """
    """
    print("Up till now we support:\n")
    for i in range(len(MODEL_NAMES)):
        print("{}. {}".format(i, MODEL_NAMES[i]))
    model_index = int(input(("We use default Decision Tree as the benchmark.\n"
                             "Please input a classifier index:\n")))

    print(("Up till now we use the following metrics to evaluate the"
           " fitted classifiers on the validation and test set.\n"))
    for i in range(len(METRICS)):
        print("{}. {}".format(i, METRICS_NAMES[i].title()))
    metric_index = int(input("Please input a metrics index:\n"))

    return model_index, metric_index


def load_features(dir_path=INPUT_DIR, test=True):
    """
    Load pre-processed feature matrices.

    """
    Xs = np.load(dir_path + 'X.npz')
    ys = np.load(dir_path + 'y.npz')

    if not test:
        X_train = Xs['train']
        y_train = ys['train']
        return X_train, y_train

    X_train, X_test = Xs['train'], Xs['test']
    y_train, y_test = ys['train'], ys['test']
    return X_train, X_test, y_train, y_test


def build_benchmark(data, metric_index):
    """
    """
    X_train, X_test, y_train, y_test = data

    benchmark = DecisionTreeClassifier(**DEFAULT_ARGS["Decision Tree"])
    benchmark.fit(X_train, y_train)
    predicted_probs = benchmark.predict_proba(X_test)[:, 1]
    benchmark_score = METRICS[metric_index](y_test, benchmark.predict(X_test))

    print("{} of the benchmark default decision tree model is {:.4f}.\n".\
          format(METRICS_NAMES[metric_index], round(benchmark_score, 4)))

    return benchmark_score


def clf_predict_proba(clf, X_test):
    """
    """
    if hasattr(clf, "predict_proba"):
        predicted_prob = clf.predict_proba(X_test)[:, 1]
    else:
        prob = clf.decision_function(X_test)
        predicted_prob = (prob - prob.min()) / (prob.max() - prob.min())

    return predicted_prob


def cross_validation(clf, skf, data, metric_index, threshold):
    """
    """
    X_train, y_train = data
    predicted_probs, scores = [], []

    for train, validation in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train], X_train[validation]
        y_tr, y_val = y_train[train], y_train[validation]

        try:
            clf.fit(X_tr, y_tr)
        except:
            return None, 0.0
        predicted_prob = clf_predict_proba(clf, X_val)
        predicted_labels = np.where(predicted_prob > threshold, 1, 0)

        predicted_probs.append(predicted_prob)
        scores.append(METRICS[metric_index](y_val, predicted_labels))

    return list(itertools.chain(*predicted_probs)), np.array(scores).mean()


def find_best_threshold(model_index, metric_index, train_data,
                        verbose=False, plot=False):
    """
    """
    model_name = MODEL_NAMES[model_index]
    metric_name = METRICS_NAMES[metric_index]
    default_args = DEFAULT_ARGS[model_name]

    clf = MODELS[model_index](**default_args)
    skf = StratifiedKFold(n_splits=5, random_state=SEED)

    if plot:
        default_probs, _ = cross_validation(clf, skf, train_data, metric_index, 0.5)
        plot_predicted_scores(default_probs)

    best_score, best_threshold = 0, None
    print("Default {}. Search Starts:".format(model_name))
    for threshold in THRESHOLDS:
        _, score = cross_validation(clf, skf, train_data, metric_index, threshold)
        if verbose:
            print("\t(Threshold: {}) the cross-validation {} is {:.4f}".\
                  format(threshold, metric_name, score))
        if score > best_score:
            best_score, best_threshold = score, threshold

    print("Search Finished: The best threshold to use is {:.4f}.\n".format(best_threshold))
    return best_threshold


def tune(model_index, metric_index, train_data, best_threshold,
         n_folds=10, verbose=False):
    """
    Use grid search and cross validation to find the best set of hyper-
    parameters.

    """
    model_name = MODEL_NAMES[model_index]
    metric_name = METRICS_NAMES[metric_index]
    params_grid = GRID_SEARCH_PARAMS[model_name]
    default_args = DEFAULT_ARGS[model_name]

    best_score, best_grid = 0, None
    params = params_grid.keys()
    skf = StratifiedKFold(n_splits=n_folds, random_state=SEED)

    print("{} with Decision Threshold {}. Search Starts:".format(model_name,
                                                                 best_threshold))
    for grid in itertools.product(*(params_grid[param] for param in params)):
        args = dict(zip(params, grid))
        clf = MODELS[model_index](**default_args, **args)
        _, grid_score = cross_validation(clf, skf, train_data, metric_index, best_threshold)

        if grid_score > best_score:
            best_score, best_grid = grid_score, args
        if verbose:
            print("\t(Parameters: {}), cross-validation {} of {:.4f}".format(args, metric_name,
                                                                             grid_score))

    print("Search Finished: The best parameters to use is {}\n".format(best_grid))
    return best_grid, best_score


def evaluate_best_model(model_index, metric_index, best_threshold, best_grid, data,
                        plot=False, verbose=True):
    """
    """
    X_train, X_test, y_train, y_test = data
    model_name = MODEL_NAMES[model_index]
    metric_name = METRICS_NAMES[metric_index]
    default_args = DEFAULT_ARGS[model_name]

    clf = MODELS[model_index](**default_args, **best_grid)
    clf.fit(X_train, y_train)

    predicted_prob = clf_predict_proba(clf, X_test)
    predicted_labels = np.where(predicted_prob > best_threshold, 1, 0)
    test_score = METRICS[metric_index](y_test, predicted_labels)
    print(("Our {} classifier reached a(n) {} of {:.4f} with a decision"
           " threshold of {} on the test set.\n").format(model_name, metric_name,
                                                       test_score, best_threshold))

    if plot:
        pos = np.count_nonzero(np.append(y_train, y_test))
        prop = pos / (len(y_test) + len(y_train))
        plot_precision_recall(predicted_prob, y_test, prop)

    return test_score


def train_evaluate(model_index, metric_index, data, train_data):
    """
    """
    metric_name = METRICS_NAMES[metric_index]
    model_name = MODEL_NAMES[model_index]

    benchmark_score = build_benchmark(data, metric_index)
    best_threshold = find_best_threshold(model_index, metric_index,
                                         train_data, verbose=True, plot=True)
    best_grid, _ = tune(model_index, metric_index, train_data, best_threshold,
                        verbose=True)
    test_score = evaluate_best_model(model_index, metric_index, best_threshold,
                                     best_grid, data, plot=True, verbose=True)

    diff = test_score - benchmark_score
    print(("{} of the tuned {} is {}, {} {} than the benchmark.\n"
           "**-------------------------------------------------------------**\n\n").\
           format(metric_name, model_name, round(test_score.mean(), 4), diff,
                  ['higher', 'lower'][int(diff <= 0)]))


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    print(("You can either choose to train with a specific configuration (input 1),"
           " or all the models by metrics we have (input 2)"))
    flag = int(input("Please input your choice:\n"))

    X_train, X_test, y_train, y_test = load_features()
    data = [X_train, X_test, y_train, y_test]
    train_data = [X_train, y_train]

    if flag == 1:
        model_index, metric_index = ask()
        train_evaluate(model_index, metric_index, data, train_data)
    else:
        for model_index in range(1, len(MODELS) + 1):
            for metric_index in range(1, len(METRICS) + 1):
                train_evaluate(model_index, metric_index, data, train_data)

    _ = input("Press any key to exit.")
