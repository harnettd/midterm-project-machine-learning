"""
Functions to help with cross-validation on a dataset that has aggregations.
These are needed to prevent data leakage from training to testing datsets.
"""
import numpy as np
import pandas as pd
from itertools import product

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import svm
from sklearn.metrics import r2_score

from engineer_features import median_by_postal_code
from utils import partition, make_index_folds


def custom_cross_validation(
    df_train: pd.DataFrame, 
    n_splits: int = 5
):
    '''creates n_splits sets of training and validation folds

    Args:
      training_data: the dataframe of features and target to be divided into folds
      n_splits: the number of sets of folds to be created

    Returns:
      A tuple of lists, where the first index is a list of the training folds, 
      and the second the corresponding validation fold

    Example:
        >>> output = custom_cross_validation(train_df, n_splits = 10)
        >>> output[0][0] # The first training fold
        >>> output[1][0] # The first validation fold
        >>> output[0][1] # The second training fold
        >>> output[1][1] # The second validation fold... etc.
    '''
    indices: list[int] = df_train.index.to_list()
    partitions: list[list] = partition(indices, n_splits)
    index_folds: list[list] = make_index_folds(partitions)

    training_folds = []
    validation_folds = []

    for idx in range(len(index_folds)):
        training_folds.append(df_train.loc[index_folds[idx][0], :])
        validation_folds.append(df_train.loc[index_folds[idx][1], :])
    
    return training_folds, validation_folds


def score(
    train_df: pd.DataFrame, 
    validate_df: pd.DataFrame,
    **kwargs
) -> float:
    """
    Return the r2_score for this training, validation fold combo.

    Add median_by_pc column.
    Scale the data with MinMaxScaler.
    Select the best 8 features using SelectKBest.
    """
    X_train = train_df.drop(columns=['sold_price'])
    y_train = train_df['sold_price']
    X_test = validate_df.drop(columns=['sold_price'])
    y_test = validate_df['sold_price']

    X_train['median_by_pc'] = median_by_postal_code(X_train, y_train)
    X_test['median_by_pc'] = median_by_postal_code(X_test, y_test)

    # Scale features.
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Select the best 8 features.
    skb = SelectKBest(f_classif, k=8)
    X_train_sc_skb = skb.fit_transform(
        X_train_sc, 
        np.ravel(y_train.to_numpy())
    )
    X_test_sc_skb = skb.transform(X_test_sc)

    # Reasonable default hyperparameter values discovered through EDA:
    C = 150_000
    gamma = 5.0
    epsilon = 1.0

    # Set hyperparameters from kwargs.
    for (k, v) in kwargs.items():
        if k == 'C':
            C = v
        if k == 'gamma':
            gamma = v
        if k == 'epsilon':
            epsilon = v
    
    model = svm.SVR(
        kernel='rbf', 
        C=C, 
        gamma=gamma, 
        epsilon=epsilon
    )
    model.fit(X_train_sc_skb, y_train)
    y_pred = model.predict(X_test_sc_skb)

    return r2_score(y_test, y_pred)


def get_best_model(trials: list[dict]) -> dict:
    """
    Return the hyperparameters and score of the best performing model.
    """
    best_model = trials[0]
    for trial in trials[1:]:
        if trial['score'] > best_model['score']:
            best_model = trial

    return best_model


def hyperparameter_search(
    training_folds, 
    validation_folds, 
    param_grid
):
    '''outputs the best combination of hyperparameter settings in the param grid, 
    given the training and validation folds

    Args:
      training_folds: the list of training fold dataframes
      validation_folds: the list of validation fold dataframes
      param_grid: the dictionary of possible hyperparameter values for the chosen model

    Returns:
      A list of the best hyperparameter settings based on the chosen metric

    Example:
        >>> param_grid = {
          'max_depth': [None, 10, 20, 30],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
          'max_features': ['sqrt', 'log2']} # for random forest
        >>> hyperparameter_search(output[0], output[1], param_grid = param_grid) 
        # assuming 'ouput' is the output of custom_cross_validation()
        [20, 5, 2, 'log2'] # hyperparams in order
    '''
    assert len(training_folds) == len(validation_folds)

    trials = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for p in product(*values):
        trial = {key: p[keys.index(key)] for key in keys}
        scores = []
        for fold in range(len(training_folds)):
            train_df = training_folds[fold]
            validate_df = validation_folds[fold]
            scores.append(score(train_df, validate_df, **trial))
        trial['score'] = sum(scores) / len(scores)
        trials.append(trial)
    
    return get_best_model(trials)


if __name__ == '__main__':
    print(__doc__)
    