"""
Generate LandfillCostsbyYear.csv from RTN shipment data.

This script processes RTN landfill shipment data and merges it with USPVDB
to create a landfill costs file with PCA and case_id information.
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def validate_merge(merged_df, shipments_df):
    """
    Validate that the merge did not create duplicate rows.
    
    Args:
        merged_df: DataFrame after merge
        shipments_df: Original shipments DataFrame
        
    Raises:
        ValueError: If duplicate rows are found after merge
    """
    if len(merged_df) > len(shipments_df):
        # Find which rows were duplicated
        duplicated_mask = merged_df.duplicated(
            subset=['Site', 'State', 'Year', 'Quarter', 'Landfill Name'], 
            keep=False
        )
        duplicated_rows = merged_df[duplicated_mask]
        
        raise ValueError(
            f"Merge resulted in duplicate rows. Found {len(merged_df)} rows "
            f"after merge but expected {len(shipments_df)}.\n"
            f"Duplicated entries:\n{duplicated_rows[['Site', 'State', 'case_id']].to_string()}"
        )


def generate_landfill_costs(
    shipments_file: str,
    uspvdb_file: str,
    output_file: str
):
    """
    Generate landfill costs file from shipments and USPVDB data.
    
    Args:
        shipments_file: Path to shipments CSV file
        uspvdb_file: Path to USPVDB Excel file
        output_file: Path to output CSV file
    """
    print(f"Reading shipments data from: {shipments_file}")
    shipments_df = pd.read_csv(shipments_file)
    
    print(f"Reading USPVDB data from: {uspvdb_file}")
    uspvdb_df = pd.read_excel(uspvdb_file)
    
    # Verify required columns exist in shipments
    required_shipments_cols = ['Site', 'State', 'Year', 'Quarter', 
                               'Landfill', 'TotalCost_$', 'Shipped_kg']
    missing_cols = set(required_shipments_cols) - set(shipments_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in shipments file: {missing_cols}")
    
    shipments_df = shipments_df.rename(columns={'Landfill': 'Landfill Name'})
    
    # Verify required columns exist in USPVDB
    required_uspvdb_cols = ['case_id', 'p_name', 'p_state', 'PCA']
    missing_cols = set(required_uspvdb_cols) - set(uspvdb_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in USPVDB file: {missing_cols}")
    
    # Rename USPVDB columns to match shipments for merge
    uspvdb_df = uspvdb_df.rename(columns={
        'p_name': 'Site',
        'p_state': 'State'
    })
    
    # Select only needed columns from USPVDB
    uspvdb_df = uspvdb_df[['Site', 'State', 'case_id', 'PCA']]
    
    print(f"Merging shipments with USPVDB on Site and State...")
    
    # Check for sites in shipments that have multiple entries in USPVDB
    for site, state in shipments_df[['Site', 'State']].drop_duplicates().values:
        matching_uspvdb = uspvdb_df[
            (uspvdb_df['Site'] == site) & 
            (uspvdb_df['State'] == state)
        ]
        if len(matching_uspvdb) > 1:
            print(f"Warning: Multiple USPVDB entries found for Site: {site}, State: {state}. This may lead to duplicate rows in the merged output.")
    
    # Merge on Site and State
    merged_df = shipments_df.merge(
        uspvdb_df,
        on=['Site', 'State'],
        how='left',
        suffixes=('', '_uspvdb'),
    )
    
    # Check for unmatched rows
    unmatched = merged_df[merged_df['case_id'].isna()]
    if len(unmatched) > 0:
        print(f"Warning: {len(unmatched)} rows could not be matched to USPVDB:")
        print(unmatched[['Site', 'State']].drop_duplicates())
    
    # Validate no duplicates were created
    # validate_merge(merged_df, shipments_df)
    
    print("Calculating costs ($/ton)...")
    # Calculate Cost: (TotalCost_$ / Shipped_kg) * 1000
    # Handle division by zero
    merged_df['Cost'] = (
        merged_df['TotalCost_$'] / merged_df['Shipped_kg'].replace(0, float('nan'))
    ) * 1000
    
    # Select and reorder columns for output
    output_df = merged_df[[
        'Year', 'Quarter', 'PCA', 'Landfill Name', 'case_id', 'Site', 'State', 'Cost'
    ]]
    
    # Sort by Year, Quarter, PCA
    output_df = output_df.sort_values(['Year', 'Quarter', 'PCA'])
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing output to: {output_file}")
    output_df.to_csv(output_file, index=False)
    
    print(f"Successfully generated {output_file}")
    print(f"Output contains {len(output_df)} rows")
    print(f"Rows with valid cost: {output_df['Cost'].notna().sum()}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate LandfillCostsbyYear.csv from RTN shipment data'
    )
    
    parser.add_argument(
        '--shipments-file',
        type=str,
        default='/Users/pghosh/SOLAR/shipments_landfill_alllandfills.csv',
        help='Path to shipments CSV file (default: /Users/pghosh/SOLAR/shipments_landfill_alllandfills.csv)'
    )
    
    parser.add_argument(
        '--uspvdb-file',
        type=str,
        default='/Users/pghosh/SOLAR/ABSiCE/USPVDB/uspvdb_v3_0_20250430_with_pca.xlsx',
        help='Path to USPVDB Excel file (default: /Users/pghosh/SOLAR/ABSiCE/USPVDB/uspvdb_v3_0_20250430_with_pca.xlsx)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='RTN/LandfillCostsbyYearAllLandfills.csv',
        help='Path to output CSV file (default: RTN/LandfillCostsbyYearAllLandfills.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.shipments_file):
        raise FileNotFoundError(f"Shipments file not found: {args.shipments_file}")
    
    if not os.path.exists(args.uspvdb_file):
        raise FileNotFoundError(f"USPVDB file not found: {args.uspvdb_file}")
    
    generate_landfill_costs(
        shipments_file=args.shipments_file,
        uspvdb_file=args.uspvdb_file,
        output_file=args.output_file
    )


if __name__ == '__main__':
    main()
