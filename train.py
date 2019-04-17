import numpy as np


INPUT_DIR = "./processed_data/"

def load_features():
    Xs = np.load(INPUT_DIR + 'X.npz')
    ys = np.load(INPUT_DIR + 'y.npz')
    X_train, X_test = Xs['train'], Xs['test']
    y_train, y_test = ys['train'], ys['test']
    return X_train, X_test, y_train, y_test