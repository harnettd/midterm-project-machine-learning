"""
Some utility functions.
"""
import pandas as pd
from numpy import ndarray, sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def print_scores(
    train: list[ndarray, ndarray, ndarray],
    test: list[ndarray, ndarray, ndarray]
) -> None:
    """
    Print various machine-learning regression test scores.

    :param train: Training data, *i.e.,* [X_train, y_train, y_train_pred]
    :type train: list[ndarray, ndarray]

    :param test: Testing data, *i.e.,* [X_test, y_test, y_test_pred]
    :type test: list[ndarray, ndarray]
    """
    X_train, y_train, y_train_pred = train
    X_test, y_test, y_test_pred = test
    
    print(f'RMSE train: {sqrt(mean_squared_error(y_train, y_train_pred))}')
    print(f'RMSE test: {sqrt(mean_squared_error(y_test, y_test_pred))}')
    print(f'MAE train: {sqrt(mean_absolute_error(y_train, y_train_pred))}')
    print(f'MAE test: {sqrt(mean_absolute_error(y_test, y_test_pred))}')
    print(f'R**2 train: {sqrt(r2_score(y_train, y_train_pred))}')
    print(f'R**2 test: {sqrt(r2_score(y_test, y_test_pred))}')


def make_coef_dict(X: pd.DataFrame, coefs: ndarray) -> dict:
    """
    Return a feature-regression coefficient dictionary.
    """
    columns = X.columns.to_list()
    fc_dict = {}
    for (feature, coef) in zip(columns, coefs):
        fc_dict[feature] = coef


if __name__ == '__main__':
    print(__doc__)
