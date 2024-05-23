"""
Some utilitey functions.
"""
import pandas as pd
from numpy import ndarray, sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def print_scores(
        y_train,
        y_train_pred,
        y_test, 
        y_test_pred
) -> None:
    """
    Print various machine-learning regression test scores.
    """
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
