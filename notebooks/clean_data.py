"""
Functions that help with data cleaning.
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


def trim_outliers(
        df: pd.DataFrame, 
        columns: list[str]
) -> pd.DataFrame:
    """
    Return a DataFrame with outliers trimmed from columns.

    Use the IQR method to trim outliers.

    :param df: A DataFrame
    :type df: pd.DataFrame

    :param old_columns: A list of columns that may contain outliers
    :param old_params: list[str]
    """
    df_subset = df[columns]

    describe = df_subset.describe() 
    first_quartile = describe.loc['25%', :]
    third_quartile = describe.loc['75%', :]
    iqr = third_quartile - first_quartile
    
    lower_bound = first_quartile - 1.5 * iqr
    upper_bound = third_quartile + 1.5 * iqr  
    lower_filter = df_subset >= lower_bound
    upper_filter = df_subset <= upper_bound
    filter = (lower_filter & upper_filter).all(axis=1)

    return df[filter]


if __name__ == '__main__':
    print(__doc__)
