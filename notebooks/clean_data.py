"""
Function that help with data cleaning.
"""
import pandas as pd


def get_nearly_empty_columns(
        df: pd.DataFrame, 
        threshold: float = 95.0 
) -> list[str]:
    """
    Return a list of columns that are nearly empty.

    :param df: A DataFrame
    :type df: pd.DataFrame

    :param threshold: The percentage threshold for a column to
        be considered empty. 
    :type threshold: float

    :return: A list of the names of nearly-empty columns
    :rtype: list[str]
    """
    num_cols: int = df.shape[0]
    num_na_by_col: pd.Series= df.isna().sum(axis='rows')
    pct_na_by_col: pd.Series = num_na_by_col / num_cols * 100
    empty_cols_filter: pd.Series = pct_na_by_col > threshold
    empty_cols: list[str] =\
        empty_cols_filter[empty_cols_filter].index.to_list()

    return empty_cols


def impute(df: pd.DataFrame, imputes: dict) -> pd.DataFrame:
    pass


def impute_with_false(ser: pd.Series) -> pd.Series:
    """
    Return a Series with each NA replaces by False.
    """
    return ser.fillna(False)


def impute_with_zero(ser: pd.Series) -> pd.Series:
    """
    Return a Series with each NA replaces by 0.
    """
    return ser.fillna(0)


def impute_with_median(ser: pd.Series) -> pd.Series:
    """
    Return a Series with each NA replaced by the Series's median.
    """
    median = ser.median()
    return ser.fillna(median)


def impute_with_mode(ser: pd.Series) -> pd.Series:
    """
    Return a Series with each NA replaced by the Series's median.
    """
    mode = ser.mode()[0]
    return ser.fillna(mode)


def impute_with_city(ser: pd.Series) -> pd.Series:
    pass


if __name__ == '__main__':
    print(__doc__)
    df = pd.read_csv('data/processed/housing_data_0.csv', sep=',')

    cols = get_nearly_empty_columns(df)
    print(df.shape, cols)
