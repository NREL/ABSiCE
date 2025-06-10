import pandas as pd
import numpy as np
from utils import TIMESTEP
import argparse
import logging

"""
Compare PCA results from two DataFrames based on a key column.
This script reads two CSV files, compares their PCA results based on a specified key column,
and outputs the differences in a new DataFrame.

To run the script, use the following command:
python compare_results.py <file1.csv> <file2.csv> --timestep <annual|monthly|quarterly>

Where:
- <file1.csv> is the path to the first CSV file (should be annual data).
- <file2.csv> is the path to the second CSV file (can be annual, monthly, or quarterly data).
- --timestep specifies the timestep of the second file (default is annual). Choices are 'annual', 'monthly', or 'quarterly'.
"""


def compare_pca_results(df1: pd.DataFrame, df2: pd.DataFrame, key: str, timestep: TIMESTEP) -> pd.DataFrame:
    """
    Compare two DataFrames based on a key column and return a DataFrame with differences.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        key (str): Column name to use as the key for comparison.

    Returns:
        pd.DataFrame: DataFrame containing the differences between df1 and df2.
    """

    diff = pd.DataFrame(columns=['year', 'pca', 'eol', 'base', 'test' 'diff'])


    for year in df1['Year'].unique():
        logger.debug(f"Processing year: {year}")
        if year not in df2['Year'].unique():
            raise ValueError(f"Year {year} not found in the second DataFrame.")
        
        if timestep == TIMESTEP.ANNUAL:
            lhs_year = df1.loc[df1['Year'] == year, key].values[0]
            rhs_year = df2.loc[df2['Year'] == year, key].values[0]
        elif timestep == TIMESTEP.MONTHLY:
            lhs_year = df1.loc[df1['Year'] == year, key].values[0]
            rhs_year = df2.loc[(df2['Year'] == year) & (df2['Month'] == 1), key].values[0]
        elif timestep == TIMESTEP.QUARTERLY:
            lhs_year = df1.loc[df1['Year'] == year, key].values[0]
            rhs_year = df2.loc[(df2['Year'] == year) & (df2['Quarter'] == 1), key].values[0]
        else:
            raise ValueError("Unsupported timestep. Use ANNUAL, MONTHLY, or QUARTERLY.")

        #convert string representations of dictionaries to actual dictionaries
        lhs_year = eval(lhs_year) if isinstance(lhs_year, str) else lhs_year
        rhs_year = eval(rhs_year) if isinstance(rhs_year, str) else rhs_year

        for pca, waste_type in lhs_year.items():
            if pca not in rhs_year:
                diff = pd.concat([
                    diff,
                    pd.DataFrame({
                        'year': year,
                        'pca': pca,
                        'eol': '',
                        'base': np.nan,
                        'test': np.nan,
                        'diff': np.nan
                    }, index=[0])
                ], ignore_index=True)
            else:
                for (eol, value) in waste_type.items():
                    if eol not in rhs_year[pca]:
                        diff = pd.concat([
                            diff,
                            pd.DataFrame({
                                'year': year,
                                'pca': pca,
                                'eol': eol,
                                'base': value,
                                'test': np.nan,
                                'diff': np.nan
                            }, index=[0])
                        ], ignore_index=True)
                    else:
                        diff_value = value - rhs_year[pca][eol]
                        if not np.isclose(diff_value, 0, atol=1e-5):
                            diff = pd.concat([
                                diff,
                                pd.DataFrame({
                                    'year': year,
                                    'pca': pca,
                                    'eol': eol,
                                    'base': value,
                                    'test': rhs_year[pca][eol],
                                    'diff': diff_value
                                }, index=[0])
                            ], ignore_index=True)

    return diff.reset_index(drop=True)

def get_timestep_from_argument(timestep: str) -> TIMESTEP:
    """
    Convert a string argument to a TIMESTEP enum.

    Args:
        timestep (str): The timestep as a string.

    Returns:
        TIMESTEP: Corresponding TIMESTEP enum.
    """
    if timestep == 'annual':
        return TIMESTEP.ANNUAL
    elif timestep == 'monthly':
        return TIMESTEP.MONTHLY
    elif timestep == 'quarterly':
        return TIMESTEP.QUARTERLY
    else:
        raise ValueError("Invalid timestep. Use 'annual', 'monthly', or 'quarterly'.")
    
def main():
    arg_parser = argparse.ArgumentParser(description="Compare PCA results from different timesteps.")
    arg_parser.add_argument('file1', type=str, help='Path to the first CSV file. Should be annual data.')
    arg_parser.add_argument('file2', type=str, help='Path to the second CSV file. Can be annual, monthly, or quarterly data.')
    arg_parser.add_argument('--timestep', '-ts', type=str, choices=['annual', 'monthly', 'quarterly'], default='annual',
                            help='Timestep of the second file. Default is annual.')
    args = arg_parser.parse_args()
    df1 = pd.read_csv(args.file1)
    df2 = pd.read_csv(args.file2)
    key = 'Waste (kg) by pca'
    differences = compare_pca_results(df1, df2, key, get_timestep_from_argument(args.timestep))
    if differences.empty:
        logger.info("No differences found between the two PCA results.")
    else:
        logger.info("Differences found between the two PCA results:")
        differences.to_csv('pca_differences.csv', index=False)
        logger.info("differences saved to 'pca_differences.csv'.")        


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    logger.info("Starting PCA results comparison.")
    main()         