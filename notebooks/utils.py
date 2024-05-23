"""
Various utility functions to help with regression analyses.
"""
import pandas as pd
from numpy import ndarray, sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def adj_r2_score(
    X: ndarray, 
    y_true: ndarray, 
    y_pred: ndarray) -> float:
    """
    Return the adjusted R-squared score.
    """
    r_squared = r2_score(y_true, y_pred)
    n, p = X.shape
    return 1 - (n - 1) / (n - p - 1) * (1 - r_squared)


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
    print(f'MAE train: {mean_absolute_error(y_train, y_train_pred)}')
    print(f'MAE test: {mean_absolute_error(y_test, y_test_pred)}')
    print(f'R**2 train: {r2_score(y_train, y_train_pred)}')
    print(f'R**2 test: {r2_score(y_test, y_test_pred)}')
    print(f'Adj R**2 train: {adj_r2_score(X_train, y_train, y_train_pred)}')
    print(f'Adj R**2 test: {adj_r2_score(X_test, y_test, y_test_pred)}')


def run_regression(
    train: [ndarray, ndarray],
    test: [ndarray, ndarray],
    regressor
):
    """
    Return a fitted regressor.

    Fit the regressor and print out a variety of test score.

    :param train: [X_train, y_train]
    :param test: [X_test, y_train]
    :param regressor: a regression model instance such as LinearRegression()

    :return: A fitted regression model
    """
    X_train, y_train = train
    X_test, y_test = test

    regressor.fit(X_train, y_train)

    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    print_scores(
        [X_train, y_train, y_train_pred], 
        [X_test, y_test, y_test_pred]
    )

    return regressor


if __name__ == '__main__':
    print(__doc__)
