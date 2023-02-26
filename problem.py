import os
import pandas as pd
import numpy as np
import rampwf as rw

from sklearn.model_selection import StratifiedShuffleSplit
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score

problem_title = 'Classification of word-level american sign language videos'

_target_column_name = 'pct_admitted_female_among_admitted'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass()
# An object implementing the workflow
workflow = rw.workflows.Estimator()

class Accuracy(BaseScoreType):
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='accuracy', precision=5):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


score_types = [
    Accuracy(name='accuracy', precision=5),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
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