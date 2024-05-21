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


def or_merge(
    df: pd.DataFrame,
    old_columns: list[str],
    new_column: str
) -> pd.DataFrame:
    """
    Return a DataFrame with columns combined to a new column using `or'.

    :param df: A DataFrame
    :type df: pd.DataFrame

    :param old_columns: A list of columns to bemerged using 'or'
    :param old_params: list[str]

    :param new_column: The name for the new, merged column
    :param type: str

    :return: A DataFrame with merged columns
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()
    
    # Generate the new column.
    df_copy[new_column] = False
    for column in old_columns:
        df_copy[new_column] = df_copy[new_column] | df_copy[column]

    # Drop the old columns.
    df_copy = df_copy.drop(columns=old_columns)
    
    return df_copy 


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
