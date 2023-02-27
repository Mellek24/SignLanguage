import os
import pandas as pd
import numpy as np
import rampwf as rw

from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

problem_title = 'Classification of word-level american sign language videos'

_target_column_name = 'pct_admitted_female_among_admitted'
# A type (class) which will be used to create wrapper objects for y_pred
_prediction_label_names = list(range(0, 2000))
Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Estimator()


class Accuracy(BaseScoreType):

    def __init__(self, name='accuracy', precision=5):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


class KTopAccuracy(BaseScoreType):

    def __init__(self, name='k_top_accuracy', precision=5, k=1):
        self.name = name
        self.precision = precision
        self.k = k

    # predictions are the probs for each class
    def __call__(self, predictions, y_true):
        sorted_indices = np.argsort(predictions, axis=1)[:, -self.k:]
        correct = np.array([y_true[i] in sorted_indices[i] for i in range(len(y_true))])
        return np.mean(correct)


score_types = [
    Accuracy(name='accuracy', precision=5),
    KTopAccuracy(name='2_top_accuracy', precision=5, k=2),
    KTopAccuracy(name='3_top_accuracy', precision=5, k=3)
]


def get_cv(X, y):
    cv = StratifiedKFold(n_splits=5, random_state=42)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)