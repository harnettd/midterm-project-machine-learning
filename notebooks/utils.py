"""
Various utility functions to help with regression analyses.
"""
import pandas as pd
from numpy import ndarray, sqrt
from numpy.random import shuffle
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


def partition(indices: list[int], n_splits: int) -> list[list]:
    """
    Partition a list into n_splits (nearly) equal-length sublists.

    :param indices: A list of integers
    :type indices: list[int]

    :return: A partition of indices (shuffled) of n_splits
    :rtype: list[list]
    """
    partitions = []
    num_per_partition = len(indices) // n_splits

    shuffle(indices)
    
    for n in range(n_splits - 1):
        start = n * num_per_partition
        stop = (n + 1) * num_per_partition
        partitions.append(indices[start:stop])
    partitions.append(indices[stop:])

    return partitions


def flatten(x: list[list]) -> list:
    """
    Return a flattened list.
    """
    y = []
    for z in x:
        y.extend(z)
    return y


def make_index_folds(index_partition: list[list]) -> list[list]:
    """
    Return a collection of train-validate folds
    """
    train_validate = []
    
    for p in range(len(index_partition)):
        tmp = index_partition.copy()
        validate = tmp.pop(p)
        train = flatten(tmp)
        train_validate.append([train, validate])

    return train_validate 


if __name__ == '__main__':
    print(__doc__)
