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


if __name__ == '__main__':
    print(__doc__)
    df = pd.read_csv('data/processed/housing_data_0.csv', sep=',')

    cols = get_nearly_empty_columns(df)
    print(df.shape, cols)
