from enum import Enum
import pandas as pd

class TIMESTEP(Enum):
    """
    Enum for time step types.
    """
    ANNUAL = 1
    MONTHLY = 12
    QUARTERLY = 4

class GeneratorSize(Enum):
    """
    Enum for different generator sizes.
    """
    VERY_SMALL = "very_small"
    SMALL = "small"
    LARGE = "large"

class ConsumerAgentResolution(Enum):
    """
    Enum for consumer agent resolution types.
    """
    PCA = "pca"
    SITE = "site"

PCA_MISSING_VALUE = "Outside ReEDS Region (No Match)"

def transform_timeseries_timestep(
    timeseries: pd.DataFrame, timestep: TIMESTEP, scale: bool = True
) -> pd.Series:
    """
    Transform a time series to the specified time step.

    Args:
        timeseries (pd.Dataframe): The input time series.
        timestep (TIMESTEP): The desired time step.
        scale (bool): Whether to scale the values by the number of periods in the timestep.

    Returns:
        pd.Series: The transformed time series.
    """
    df = timeseries.copy()
    df_columns = df.select_dtypes(include=["number"]).columns.tolist()
    df_columns = [col for col in df_columns if col not in ["year"]]
    df["date"] = pd.to_datetime(df["year"].astype(str))
    if timestep == TIMESTEP.ANNUAL:
        return df
    
    df.set_index("date", inplace=True)
    start = df.index.min()
    end = pd.to_datetime(f"{df.index.max().year}-12-31")

    if timestep == TIMESTEP.MONTHLY:
        freq = "MS"
        periods = 12
    elif timestep == TIMESTEP.QUARTERLY:
        freq = "QS"
        periods = 4
    else:
        raise ValueError("Unsupported timestep. Use ANNUAL, MONTHLY, or QUARTERLY.")
    all_dates = pd.date_range(start=start, end=end, freq=freq)
    df_expanded = df.reindex(all_dates, method='ffill')
    if timestep == TIMESTEP.MONTHLY:
        df_expanded['month'] = df_expanded.index.month
    elif timestep == TIMESTEP.QUARTERLY:
        df_expanded['quarter'] = df_expanded.index.quarter
    if scale:
        for col in set(df_columns):
            df_expanded[col] = df_expanded[col]/ periods
    df_expanded.rename_axis('date', inplace=True)
    df_expanded = df_expanded.reset_index()

    return df_expanded
    
def transform_pca_timeseries_timestep(
    timeseries: pd.DataFrame, timestep: TIMESTEP, filtered_columns: list = None
) -> pd.DataFrame:
    """
    Transform a PCA time series to the specified time step.

    Args:
        timeseries (pd.DataFrame): The input PCA time series.
        timestep (TIMESTEP): The desired time step.

    Returns:
        pd.DataFrame: The transformed PCA time series.
    """
    df = timeseries.copy()
    
    if timestep == TIMESTEP.ANNUAL:
        return timeseries
    elif timestep == TIMESTEP.MONTHLY:
        freq = "MS"
        periods = 12
    elif timestep == TIMESTEP.QUARTERLY:
        freq = "QS"
        periods = 4
    else:
        raise ValueError("Unsupported timestep. Use ANNUAL, MONTHLY, or QUARTERLY.")
    
    # if filtered_columns is provided, filter the DataFrame
    if filtered_columns is not None:
        df = df[filtered_columns + ['year', 'pca']]
    
    # select all numeric columns except 'year'
    df_columns = df.select_dtypes(include=["number"]).columns.tolist()
    df_columns = [col for col in df_columns if col not in ["year"]]

    # get the starting offset depending on the time step, then create a list of offsets
    offset = pd.offsets.MonthBegin(1) if timestep == TIMESTEP.MONTHLY else pd.offsets.QuarterBegin(startingMonth=1)
    time_offsets = [i * offset for i in range(periods)]


    # create a DataFrame with the time offsets
    time_offsets_df = pd.DataFrame({'offset': time_offsets})

    df['base_date'] = pd.to_datetime(df['year'].astype(str))

    # create a cross join to expand the DataFrame with all combinations of base_date and offsets
    df_expanded = df.merge(time_offsets_df, how='cross')
    df_expanded['date'] = df_expanded['base_date'] + df_expanded['offset']
    df_expanded['date'] = pd.to_datetime(df_expanded['date'])

    if timestep == TIMESTEP.MONTHLY:
        df_expanded['month'] = df_expanded['date'].dt.month
    elif timestep == TIMESTEP.QUARTERLY:
        df_expanded['quarter'] = df_expanded['date'].dt.quarter

    df_expanded[df_columns] = df_expanded[df_columns].div(periods)
    df_expanded = df_expanded.sort_values(by='date').reset_index(drop=True)
    df_expanded = df_expanded.drop(columns=['offset'])

    return df_expanded
