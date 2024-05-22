"""
Functions related to feature engineering.
"""
import pandas as pd


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


def month_to_season(month: int) -> str:
    """
    Return the season corresponding to a month.

    Assumes that spring is Mar-May, summer is Jun-Aug,
    autumn is Sept-Nov, and winter is Dec-Feb.

    :param month: A month as an integer from 1-12
    :type month: int

    :return: The season corrsponding to month
    :rtype: str
    """
    assert month in list(range(1, 13))

    months_of_seasons = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2]
    }
    for season in months_of_seasons:
        if month in months_of_seasons[season]:
            return season


def get_seasons(dates: pd.Series) -> pd.Series:
    """
    Return a Series whose elements are seasons.

    :param dates: A series of dates
    :type dates: pd.Series

    :return: A series of seasons
    :rtype: pd.Series
    """
    return dates.dt.month.apply(month_to_season)


def median_by_postal_code(
        df: pd.DataFrame, 
        sold_prices: pd.Series
) -> pd.Series:
    """
    Return a Series of median sale prices by postal code.

    :param df: A DataFrame with 'postal_code' as a column
    :type df: pd.DataFrame

    :param sold_prices: A Series of home sale prices
    :type sold_prices: pd.Series

    :return: A Series of median sales prices by postal code
    :rtype: pd.Series
    """
    assert 'postal_code' in df.columns.to_list()
    assert sold_prices.name == 'sold_price'

    # Construct a DataFrame with 'sold_price' and 'postal_code' columns
    sp_and_pc = pd.DataFrame([sold_prices, df['postal_code']]).T
    sp_and_pc['postal_code'] = sp_and_pc['postal_code'].astype('int')

    # Compute all median sale prices by postal code.
    median_sp_by_pc = sp_and_pc.groupby(by='postal_code').median()

    return df['postal_code'].apply(lambda pc: median_sp_by_pc.loc[pc])


if __name__ == '__main__':
    print(__doc__)
