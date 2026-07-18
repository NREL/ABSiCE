import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import ast
import os
import argparse
import re
import geopandas as gpd
import pyproj
pyproj.network.set_network_enabled(False)

def plot_waste_by_pca(
        results_dir="results"  # Directory containing the results CSV files
):
    """
    Plot the waste data by PCA components.

    Args:
        results_dir (str): Path to the directory containing the results CSV files.  
    """
    
    
    df_plot = pd.DataFrame()
    for file in os.listdir(results_dir):
        #grab all files that match the pattern Results_model_run_{integer}.csv
        if re.match(r"Results_model_run_\d+.csv", file):
            print(f"Processing file: {file}")
            file_path = os.path.join(results_dir, file)
            waste_df = pd.read_csv(file_path, index_col=0)
            number = int(file.split('_')[-1].split('.')[0])

            for year in waste_df['Year'].unique():
                year_data = waste_df[waste_df['Year'] == year]
                pca_waste_dict = year_data['Waste (kg) by pca'].values[0]
                pca_waste_dict = eval(pca_waste_dict) 
                for pca, waste_dict in pca_waste_dict.items():
                    pca_df = pd.DataFrame({
                        'Year': year,
                        'PCA': pca,
                        "Run": number,
                    },
                    index=[0])
                    for waste_type, waste in waste_dict.items():
                        waste = float(waste) if isinstance(waste, str) else waste
                        pca_df[waste_type] = waste

                    df_plot = pd.concat([
                        df_plot,
                        pca_df
                    ], ignore_index=True)
    
    # Get the average of each PCA component across all run
    df_plot = df_plot.groupby(['Year', 'PCA']).mean(numeric_only=True).reset_index()
    df_plot = df_plot[['Year', 'PCA'] + [col for col in df_plot.columns if col not in ['Year', 'PCA', 'Run']]]
    df_plot = df_plot.loc[df_plot['PCA'].isin(['p27','p28','p29','p30','p48','p57','p59','p60','p61','p62','p63','p64','p65','p66','p67'])]

    # df_plot_5_years = df_plot[(df_plot['Year'] > 2025) & (df_plot['Year'] <= 2030)]
    df_plot_5_years = df_plot.copy()
    df_plot_5_years.to_csv(os.path.join(results_dir, "waste_kg_per_year_by_pca.csv"), index=False) 
    
    # Plotting
    plt.figure(figsize=(12, 6))

    df_plot = df_plot.sort_values(by=["Year", "PCA"])
    for waste_type in df_plot.columns:
        if waste_type not in ['Year', 'PCA', 'Waste (kg)']: 
            for pca in df_plot['PCA'].unique():
                pca_data = df_plot[df_plot['PCA'] == pca]
                plt.plot(
                    pca_data['Year'],
                    pca_data[waste_type],
                    marker='o',
                    label=pca,
                )
            
            plt.title(f'Waste by PCA Components - {waste_type}')
            plt.xlabel('Year')
            plt.ylabel('Waste (kg)')
            plt.xticks(df_plot['Year'].unique(), rotation=45)
            plt.legend(title='PCA Components')
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"waste_by_pca_{waste_type}.png"))
            plt.close()

def _aggregate_results(results_dir: str = "results") -> pd.DataFrame:
    """
    Aggregate results from multiple runs.

    Args:
        results_dir (str): Path to the directory containing the results CSV files.
    """
    df_aggregate = pd.DataFrame()

    for file in os.listdir(results_dir):
        #grab all files that match the pattern Results_agents_consumers_run_{integer}.csv
        if re.match(r"Results_agents_consumers_run_\d+.csv", file):
            print(f"Processing file: {file}")
            file_path = os.path.join(results_dir, file)
            waste_df = pd.read_csv(file_path)
            number = int(file.split('_')[-1].split('.')[0])
            waste_df['Run'] = number
            df_aggregate = pd.concat([df_aggregate, waste_df], ignore_index=True)
    
    return df_aggregate

def aggregate_and_save_consumer_results(results_dir: str = "results", columns: list[str] = []) -> None:
    """
    Aggregate consumer results from multiple runs and save to a CSV file.

    Args:
        results_dir (str): Path to the directory containing the results CSV files.
    """
    df_aggregate = _aggregate_results(results_dir=results_dir)
    
    # take the average of each consumer across all runs
    if "Quarter" in df_aggregate.columns:
        index_cols = ['Year', 'Quarter', 'PCA', 'State', 'Name', 'Latitude', 'Longitude']
    else:
        index_cols = ['Year', 'PCA', 'State', 'Name', 'Latitude', 'Longitude']
    df_aggregate = df_aggregate.groupby(index_cols).mean(numeric_only=True).reset_index()
    if columns:
        df_aggregate = df_aggregate[index_cols + columns]
    df_aggregate_5_years = df_aggregate[(df_aggregate['Year'] > 2025) & (df_aggregate['Year'] <= 2040)]
    df_aggregate_5_years.to_csv(os.path.join(results_dir, "waste_kg_per_year_by_site.csv"), index=False)

def plot_consumer_waste_by_site(
        results_dir: str = "results",  # Directory containing the results CSV files
        value_columns: list[str] = []
):
    """
    Plot the consumer waste data by site.

    Args:
        results_dir (str): Path to the directory containing the results CSV files.  
        value_columns (list[str]): List of columns to aggregate and plot.
    """
    
    states = gpd.read_file(os.path.join("tl_2025_us_state","tl_2025_us_state.shp"))
    # contiguous United States (excluding Alaska, Hawaii, Puerto Rico, Virgin Islands, Guam, Northern Mariana Islands, and American Samoa)
    states = states[~states['STUSPS'].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS'])]
    states = states.to_crs(epsg=5070)  # Albers Equal Area
    
    df = pd.read_csv(os.path.join(results_dir, "waste_kg_per_year_by_site.csv"))
    df_sum = df.groupby(['Name', 'Latitude', 'Longitude'], as_index=False).sum(numeric_only=True).reset_index()
    df_sum["Total Waste (kg)"] = df_sum[value_columns].sum(axis=1)
    gdf = gpd.GeoDataFrame(
        df_sum, geometry=gpd.points_from_xy(df_sum.Longitude, df_sum.Latitude, crs="EPSG:4326")
    )

    gdf = gdf.to_crs(states.crs)  # Albers Equal Area

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    states.boundary.plot(ax=ax, color='black', linewidth=0.5)
    # plot "Total Waste (kg)" in map at each site location with size proportional to the log of the waste amount
    gdf.plot(
        ax=ax,
        column="Total Waste (kg)",
        cmap="Reds",
        markersize=np.sqrt(gdf["Total Waste (kg)"]) * 0.002,  # scale marker size
        legend=True,
        # legend_kwds={'label': "Total Waste (kg)", 'shrink': 0.7},
        alpha=0.7,
        edgecolor='k',
        label='PV Waste Sites',
        cax=cax
    )

    cax.set_ylabel('Total Waste (kg)', fontsize=50)
    # print("cax ticks:", cax.get_yticks())
    # cax.set_yticks(cax.get_yticks(), fontsize=16)
    cax.set_yticklabels(cax.get_yticklabels(), fontsize=30)

    # Format colorbar with scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # Use scientific notation for values outside this range
    cax.yaxis.set_major_formatter(formatter)
    cax.yaxis.offsetText.set_fontsize(30)

    # ax.set_title('Total Waste by Site (kg)', fontsize=16)
    ax.set_axis_off() # Hide axis
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "consumer_waste_by_site.jpg"), bbox_inches='tight')
    plt.close()

def aggregate_and_plot_consumer_tclp_results(
        results_dir: str = "results",  # Directory containing the results CSV files
):
    """
    Aggregate and plot consumer TCLP results.

    Args:
        results_dir (str): Path to the directory containing the results CSV files.  
    """

    df_aggregate = _aggregate_results(results_dir=results_dir)
    
    assert "TCLP Test Result" in df_aggregate.columns, "TCLP Test Result column not found in the data."

    # Create box and whisker plot for TCLP Test Result success proportion by Year
    plt.figure(figsize=(10, 6))
    df_aggregate = df_aggregate.groupby(['Year', 'Run']).mean(numeric_only=True).reset_index()
    df_aggregate['TCLP Test Result'] = df_aggregate['TCLP Test Result'].astype(float)
    # limit years to 2025-2040
    df_aggregate = df_aggregate[(df_aggregate['Year'] >= 2025) & (df_aggregate['Year'] <= 2040)]
    df_aggregate.boxplot(column='TCLP Test Result', by='Year')
    plt.title('')
    plt.suptitle('')
    plt.xlabel('Year', fontsize=20)
    # set xticks rotation and size
    plt.xticks(rotation=45, fontsize=16)
    plt.ylabel('Hazardous Proportion', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "consumer_tclp_results_by_year.jpg"), bbox_inches='tight')
    plt.close()


def _calculate_waste_rates(
        df: pd.DataFrame,
        waste_columns: list[str],
        group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate waste-management rates using summed masses.

    This implements the same formula used by sensitivity plots:
    rate = sum(component waste) / sum(total waste).

    Parameters:
    df (pd.DataFrame): Input dataframe containing waste columns.
    waste_columns (list[str]): Waste mass columns in kg.
    group_cols (list[str] | None): Optional grouping columns. If provided,
        rates are calculated per group; otherwise one overall row is returned.
    Returns:
    pd.DataFrame: Dataframe containing group columns (if any) and rate columns.
    """
    if group_cols:
        summed: pd.DataFrame = df.groupby(group_cols)[waste_columns].sum().reset_index()
    else:
        summed = pd.DataFrame([df[waste_columns].sum(numeric_only=True)])

    summed["Total Waste (Kg)"] = summed[waste_columns].sum(axis=1)

    rate_columns: list[str] = []
    for col in waste_columns:
        rate_col: str = col.replace('(Kg)', 'Rate')
        rate_columns.append(rate_col)
        summed[rate_col] = (summed[col] / summed["Total Waste (Kg)"]).fillna(0)

    output_cols: list[str] = (group_cols or []) + rate_columns
    return summed[output_cols]


def _prepare_compare_waste_management_data(
        results_dir1: str,
        results_dir2: str,
        use_sum_method: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, float]]]:
    """
    Prepare grouped and overall waste-management rates for scenario comparison.

    Parameters:
    results_dir1 (str): Path to first results directory.
    results_dir2 (str): Path to second results directory.
    use_sum_method (bool): If True, use summed-mass rates by year/quarter.
        If False, average site-level rates by year/quarter.
    Returns:
    tuple[pd.DataFrame, list[str], dict[str, dict[str, float]]]: Combined
        year-quarter rates, rate column names, and scenario-wide overall rates.
    """
    df1 = pd.read_csv(os.path.join(results_dir1, "waste_kg_per_year_by_site.csv"))
    df2 = pd.read_csv(os.path.join(results_dir2, "waste_kg_per_year_by_site.csv"))

    df1 = df1[(df1['Year'] >= 2026) & (df1['Year'] <= 2030)]
    df2 = df2[(df2['Year'] >= 2026) & (df2['Year'] <= 2030)]

    waste_columns = ['Waste Repair (Kg)', 'Waste Sell (Kg)', 'Waste Recycle (Kg)',
                     'Waste Landfill (Kg)', 'Waste Hoard (Kg)']
    rate_columns = [col.replace('(Kg)', 'Rate') for col in waste_columns]

    if use_sum_method:
        df1_rates = _calculate_waste_rates(df1, waste_columns, group_cols=['Year', 'Quarter'])
        df2_rates = _calculate_waste_rates(df2, waste_columns, group_cols=['Year', 'Quarter'])
    else:
        def calculate_rates(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df['Total Waste'] = df[waste_columns].sum(axis=1)
            for col in waste_columns:
                rate_col = col.replace('(Kg)', 'Rate')
                df[rate_col] = (df[col] / df['Total Waste']).fillna(0)
            return df

        df1_rates = calculate_rates(df1).groupby(['Year', 'Quarter'])[rate_columns].mean().reset_index()
        df2_rates = calculate_rates(df2).groupby(['Year', 'Quarter'])[rate_columns].mean().reset_index()

    df1_rates = df1_rates[['Year', 'Quarter'] + rate_columns].copy()
    df1_rates['Scenario'] = 'All Landfills'

    df2_rates = df2_rates[['Year', 'Quarter'] + rate_columns].copy()
    df2_rates['Scenario'] = 'True Landfills'

    df_combined = pd.concat([df1_rates, df2_rates], ignore_index=True)
    df_combined['Year-Quarter'] = (
        df_combined['Year'].astype(str) + '-Q' + df_combined['Quarter'].astype(str)
    )

    overall_rates_map: dict[str, dict[str, float]] = {
        'All Landfills': _calculate_waste_rates(df1, waste_columns).iloc[0].to_dict(),
        'True Landfills': _calculate_waste_rates(df2, waste_columns).iloc[0].to_dict(),
    }
    return df_combined, rate_columns, overall_rates_map

def compare_waste_management_rates(
        results_dir1: str = "results/RTN_run_all_landfills",
        results_dir2: str = "results/RTN_run_true_landfills",
        output_dir: str = "results",
        use_sum_method: bool = True
):
    """
    Compare and plot average waste management rates by year and quarter for 2026-2030.
    
    Args:
        results_dir1 (str): Path to first results directory (all landfills scenario).
        results_dir2 (str): Path to second results directory (true landfills scenario).
        output_dir (str): Directory to save output plots.
        use_sum_method (bool): If True, sum waste amounts then calculate rates. 
                               If False, calculate rates per site then average. Default is True.
    """
    
    df_combined, rate_columns, overall_rates_map = _prepare_compare_waste_management_data(
        results_dir1=results_dir1,
        results_dir2=results_dir2,
        use_sum_method=use_sum_method,
    )
    
    # Plot each waste management option
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    colors = {'All Landfills': '#1f77b4', 'True Landfills': '#ff7f0e'}
    
    for idx, rate_col in enumerate(rate_columns):
        ax = axes[idx]
        
        for scenario in ['All Landfills', 'True Landfills']:
            scenario_data = df_combined[df_combined['Scenario'] == scenario]
            ax.plot(
                scenario_data['Year-Quarter'],
                scenario_data[rate_col],
                marker='o',
                label=scenario,
                color=colors[scenario],
                linewidth=2,
                markersize=6
            )
            
            # Plot scenario-wide average using global summed-mass ratio.
            overall_avg = overall_rates_map[scenario][rate_col]
            ax.axhline(
                y=overall_avg,
                linestyle='--',
                color=colors[scenario],
                linewidth=1.5,
                alpha=0.5,
                label=f'{scenario} Avg'
            )
        
        # Clean up the title
        waste_type = rate_col.replace('Waste ', '').replace(' Rate', '')
        ax.set_title(f'{waste_type} Rate', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year-Quarter', fontsize=10)
        ax.set_ylabel('Average Rate', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Remove the extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "waste_management_rates_comparison.jpg"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create a summary plot showing all rates stacked
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, scenario in zip([ax1, ax2], ['All Landfills', 'True Landfills']):
        scenario_data = df_combined[df_combined['Scenario'] == scenario]
        
        # Create stacked bar chart
        x = range(len(scenario_data))
        x_labels = scenario_data['Year-Quarter'].values
        
        colors_stack = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        bottom = np.zeros(len(scenario_data))
        
        for idx, rate_col in enumerate(rate_columns):
            waste_type = rate_col.replace('Waste ', '').replace(' Rate', '')
            ax.bar(
                x,
                scenario_data[rate_col].values,
                bottom=bottom,
                label=waste_type,
                alpha=0.8,
                color=colors_stack[idx],
                width=0.8
            )
            bottom += scenario_data[rate_col].values
        
        ax.set_title(f'{scenario} - Waste Management Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year-Quarter', fontsize=10)
        ax.set_ylabel('Rate (Proportion)', fontsize=10)
        ax.set_xticks(x[::4])  # Show every 4th label to avoid crowding
        ax.set_xticklabels(x_labels[::4], rotation=45)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.02)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "waste_management_distribution_comparison.jpg"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    print(f"  - waste_management_rates_comparison.jpg")
    print(f"  - waste_management_distribution_comparison.jpg")


def plot_recycling_rate_comparison(
        results_dir1: str = "results/RTN_run_all_landfills",
        results_dir2: str = "results/RTN_run_true_landfills",
        output_dir: str = "results",
        use_sum_method: bool = True,
        y_axis_min: float | None = 0.0,
        y_axis_max: float | None = 0.4,
) -> None:
    """
    Plot only the recycling-rate comparison by year-quarter.

    Parameters:
    results_dir1 (str): Path to first results directory.
    results_dir2 (str): Path to second results directory.
    output_dir (str): Directory to save the output plot.
    use_sum_method (bool): If True, use summed-mass rates by year/quarter.
    y_axis_min (float | None): Optional lower y-axis limit.
    y_axis_max (float | None): Optional upper y-axis limit.
    Returns:
    None
    """
    df_combined, _, overall_rates_map = _prepare_compare_waste_management_data(
        results_dir1=results_dir1,
        results_dir2=results_dir2,
        use_sum_method=use_sum_method,
    )

    rate_col: str = 'Waste Recycle Rate'
    colors = {'All Landfills': '#1f77b4', 'True Landfills': '#ff7f0e'}

    fig, ax = plt.subplots(figsize=(12, 6))

    for scenario in ['All Landfills', 'True Landfills']:
        scenario_data = df_combined[df_combined['Scenario'] == scenario]
        ax.plot(
            scenario_data['Year-Quarter'],
            scenario_data[rate_col],
            marker='o',
            label=scenario,
            color=colors[scenario],
            linewidth=2,
            markersize=6,
        )

        overall_avg = overall_rates_map[scenario][rate_col]
        ax.axhline(
            y=overall_avg,
            linestyle='--',
            color=colors[scenario],
            linewidth=1.5,
            alpha=0.5,
            label=f'{scenario} Avg',
        )

    if y_axis_min is not None or y_axis_max is not None:
        ax.set_ylim(
            bottom=y_axis_min if y_axis_min is not None else None,
            top=y_axis_max if y_axis_max is not None else None,
        )

    ax.set_title('Recycling Rate Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year-Quarter', fontsize=10)
    ax.set_ylabel('Recycling Rate', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'recycling_rate_comparison.jpg')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_statewise_waste_management_rates(
        results_dir: str = "results",
        output_dir: str = "",
        year_range: tuple[int, int] | None = None
) -> None:
    """
    Plot statewise landfill and recycling rates for each year/quarter.

    Reads the aggregated waste_kg_per_year_by_site.csv, groups by State and
    Year/Quarter, sums waste amounts, computes landfill and recycling rates,
    then saves a heatmap (states x time) as a JPEG.

    Parameters:
    results_dir (str): Directory containing waste_kg_per_year_by_site.csv.
    output_dir (str): Directory to save the output plot. Defaults to results_dir.
    year_range (tuple[int, int] | None): Optional (start_year, end_year) inclusive
        filter. If None, all years in the file are used.
    Returns:
    None
    """
    if not output_dir:
        output_dir = results_dir

    df: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "waste_kg_per_year_by_site.csv")
    )

    if year_range is not None:
        df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

    waste_columns: list[str] = [
        "Waste Repair (Kg)",
        "Waste Sell (Kg)",
        "Waste Recycle (Kg)",
        "Waste Landfill (Kg)",
        "Waste Hoard (Kg)",
    ]

    # Determine grouping columns based on available temporal resolution
    has_quarter: bool = "Quarter" in df.columns
    group_cols: list[str] = ["State", "Year", "Quarter"] if has_quarter else ["State", "Year"]

    df_state: pd.DataFrame = df.groupby(group_cols)[waste_columns].sum().reset_index()

    df_state["Total Waste (Kg)"] = df_state[waste_columns].sum(axis=1)

    df_state["Recycling Rate"] = (
        df_state["Waste Recycle (Kg)"] / df_state["Total Waste (Kg)"]
    ).fillna(0)
    df_state["Landfill Rate"] = (
        df_state["Waste Landfill (Kg)"] / df_state["Total Waste (Kg)"]
    ).fillna(0)

    if has_quarter:
        df_state["Year-Quarter"] = (
            df_state["Year"].astype(str) + "-Q" + df_state["Quarter"].astype(str)
        )
        time_col: str = "Year-Quarter"
    else:
        time_col = "Year"

    # Sort time periods chronologically
    time_order: list = sorted(
        df_state[time_col].unique(),
        key=lambda x: str(x)
    )
    state_order: list[str] = sorted(df_state["State"].unique())

    def _pivot_rate(rate_col: str) -> pd.DataFrame:
        """
        Pivot the rate column into a (state x time) matrix.

        Parameters:
        rate_col (str): Name of the rate column to pivot.
        Returns:
        pd.DataFrame: Pivoted DataFrame with states as rows and time periods as columns.
        """
        pivot: pd.DataFrame = df_state.pivot_table(
            index="State", columns=time_col, values=rate_col, aggfunc="mean"
        )
        pivot = pivot.reindex(index=state_order, columns=time_order)
        return pivot

    recycle_pivot: pd.DataFrame = _pivot_rate("Recycling Rate")
    landfill_pivot: pd.DataFrame = _pivot_rate("Landfill Rate")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(16, len(time_order) * 0.55), max(6, len(state_order) * 0.45))
    )

    heatmap_configs: list[dict] = [
        {
            "data": recycle_pivot,
            "title": "Recycling Rate by State",
            "cmap": "coolwarm",
            "ax": axes[0],
        },
        {
            "data": landfill_pivot,
            "title": "Landfill Rate by State",
            "cmap": "coolwarm",
            "ax": axes[1],
        },
    ]

    for cfg in heatmap_configs:
        ax: plt.Axes = cfg["ax"]
        data: pd.DataFrame = cfg["data"]
        im = ax.imshow(
            data.values,
            aspect="auto",
            cmap=cfg["cmap"],
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Rate", fontsize=10)
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )

        ax.set_xticks(range(len(time_order)))
        ax.set_xticklabels(time_order, rotation=90, fontsize=8)
        ax.set_yticks(range(len(state_order)))
        ax.set_yticklabels(state_order, fontsize=9)
        ax.set_title(cfg["title"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Year-Quarter" if has_quarter else "Year", fontsize=10)
        ax.set_ylabel("State", fontsize=10)

        # Annotate cells with the rate value
        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):
                cell_val: float = data.values[row_idx, col_idx]
                if not np.isnan(cell_val):
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{cell_val:.0%}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if 0.2 < cell_val < 0.8 else "white",
                    )

    plt.tight_layout()
    output_path: str = os.path.join(output_dir, "statewise_waste_management_rates.jpg")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_total_waste_by_state(
        results_dir: str = "results",
        output_dir: str = "",
        year_range: tuple[int, int] | None = None,
        shapefile_dir: str = "tl_2025_us_state",
) -> None:
    """
    Plot total waste mass (kg) by state as a choropleth map.

    Reads the aggregated waste_kg_per_year_by_site.csv, sums all waste columns
    across years and sites per state, then fills each state polygon with a color
    proportional to the total waste produced using a diverging colormap.

    Parameters:
    results_dir (str): Directory containing waste_kg_per_year_by_site.csv.
    output_dir (str): Directory to save the output plot. Defaults to results_dir.
    year_range (tuple[int, int] | None): Optional (start_year, end_year) inclusive
        filter. If None, all years in the file are used.
    shapefile_dir (str): Directory containing the tl_2025_us_state shapefile.
    Returns:
    None
    """
    if not output_dir:
        output_dir = results_dir

    waste_columns: list[str] = [
        "Waste Repair (Kg)",
        "Waste Sell (Kg)",
        "Waste Recycle (Kg)",
        "Waste Landfill (Kg)",
        "Waste Hoard (Kg)",
    ]

    df: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "waste_kg_per_year_by_site.csv")
    )

    if year_range is not None:
        df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

    # Sum all waste types by state across all years/sites, convert kg → metric tons
    df_state: pd.DataFrame = df.groupby("State")[waste_columns].sum().reset_index()
    df_state["Total Waste (metric tons)"] = df_state[waste_columns].sum(axis=1) / 1_000

    # Load and prepare shapefile (contiguous US only)
    states: gpd.GeoDataFrame = gpd.read_file(
        os.path.join(shapefile_dir, "tl_2025_us_state.shp")
    )
    states = states[~states["STUSPS"].isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"])]
    states = states.to_crs(epsg=5070)  # Albers Equal Area

    # Merge waste totals onto shapefile
    states_merged: gpd.GeoDataFrame = states.merge(
        df_state[["State", "Total Waste (metric tons)"]],
        left_on="STUSPS",
        right_on="State",
        how="left",
    )

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    # States with no data: outline only, no fill
    states_merged[states_merged["Total Waste (metric tons)"].isna()].plot(
        ax=ax, color="none", edgecolor="black", linewidth=0.5
    )

    states_merged[states_merged["Total Waste (metric tons)"].notna()].plot(
        ax=ax,
        column="Total Waste (metric tons)",
        cmap="YlOrRd",
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        cax=cax,
        legend_kwds={"label": "Total Waste (metric tons)"},
    )

    # Format colorbar with comma-separated thousands
    cax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    cax.set_ylabel("Total Waste (metric tons)", fontsize=13)
    cax.tick_params(labelsize=11)

    # Annotate each state with its abbreviation and value
    for _, row in states_merged.iterrows():
        centroid = row.geometry.centroid
        value: float = row["Total Waste (metric tons)"]
        label: str = (
            f"{row['STUSPS']}\n{value:,.0f}" if not pd.isna(value) else row["STUSPS"]
        )
        ax.annotate(
            label,
            xy=(centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
        )

    ax.set_axis_off()
    plt.tight_layout()

    output_path: str = os.path.join(output_dir, "total_waste_by_state.jpg")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_recycling_rate_sensitivity(
        iteration_dir: str = "results/RTN_Iteration_3",
        output_dir: str = "",
        year_range: tuple[int, int] | None = None,
        sensitivity_type: str = "recycling",
        x_axis: str = "pct",
    baseline_cost_per_kg: float = 0.4,
        kg_per_w: float = 0.0077,
        w_per_module: float = 270.0,
) -> None:
    """
    Sensitivity analysis: recycling rate vs. cost delta.

    Scans subfolders of iteration_dir for all_landfills and true_landfills
    scenario results, extracts the cost ratio from the folder name suffix,
    computes the overall recycling rate (sum Waste Recycle / sum Total Waste)
    across all sites and years, and saves a CSV and line plot with one line
    per landfill set.

    For ratio == 1.0 the un-suffixed folder (e.g. RTN_run_all_landfills) is
    used as baseline for both sensitivity types.

    Parameters:
    iteration_dir (str): Base directory containing scenario subfolders.
    output_dir (str): Directory to save outputs. Defaults to iteration_dir.
    year_range (tuple[int, int] | None): Optional (start_year, end_year)
        inclusive filter on the Year column.
    sensitivity_type (str): 'recycling' (default) scans folders with plain
        ratio suffixes (e.g. _1.05, _neg0.75); 'transport' scans folders with
        _transport_ infix (e.g. _transport_1.05).
    x_axis (str): 'pct' (default) plots cost change as a percentage; 
        'cost_per_module' plots the absolute cost per module in USD.
    baseline_cost_per_kg (float): Baseline recycling cost in $/kg (default 0.4).
    kg_per_w (float): Average module mass in kg/W (default 0.0077).
    w_per_module (float): Average module wattage in W (default 270).
    Returns:
    None
    """
    if not output_dir:
        output_dir = iteration_dir

    waste_columns: list[str] = [
        "Waste Repair (Kg)",
        "Waste Sell (Kg)",
        "Waste Recycle (Kg)",
        "Waste Landfill (Kg)",
        "Waste Hoard (Kg)",
    ]

    _SET_PREFIXES: dict[str, str] = {
        "All Landfills": "RTN_run_all_landfills",
        "True Landfills": "RTN_run_true_landfills",
    }

    def _parse_ratio(folder_name: str, prefix: str) -> float | None:
        """
        Parse the cost ratio encoded in a scenario folder name.

        For sensitivity_type == 'recycling', accepts plain ratio suffixes
        (e.g. _1.05, _neg0.75) and rejects _transport_ folders.
        For sensitivity_type == 'transport', accepts _transport_ suffixes
        (e.g. _transport_1.05) and rejects plain-ratio folders.
        The un-suffixed baseline folder always returns 1.0 for both types.

        Parameters:
        folder_name (str): Full folder name.
        prefix (str): The landfill-set prefix to strip.
        Returns:
        float | None: Parsed ratio, or None if the folder should be skipped.
        """
        suffix: str = folder_name[len(prefix):]
        if not suffix:
            return 1.0  # un-suffixed = baseline for both types
        s: str = suffix.lstrip("_")
        if not s:
            return None
        if sensitivity_type == "transport":
            if not s.startswith("transport_"):
                return None  # skip non-transport folders
            s = s[len("transport_"):]
        else:  # recycling
            if s.startswith("transport"):
                return None  # skip transport-cost folders
        try:
            if s.startswith("neg"):
                return float(s[3:])
            return float(s)
        except ValueError:
            return None

    records: list[dict] = []

    for label, prefix in _SET_PREFIXES.items():
        # Track which ratios have already been recorded for this set so that
        # the un-suffixed baseline (sorted first) wins over the '_1' folder.
        seen_ratios: set[float] = set()

        for entry in sorted(os.scandir(iteration_dir), key=lambda e: e.name):
            if not entry.is_dir() or not entry.name.startswith(prefix):
                continue

            ratio: float | None = _parse_ratio(entry.name, prefix)
            if ratio is None:
                continue

            if ratio in seen_ratios:
                continue  # prefer the first (un-suffixed) folder for ratio=1.0
            seen_ratios.add(ratio)

            site_csv: str = os.path.join(entry.path, "waste_kg_per_year_by_site.csv")
            if not os.path.isfile(site_csv):
                print(f"  Skipping {entry.name}: waste_kg_per_year_by_site.csv not found")
                continue

            df: pd.DataFrame = pd.read_csv(site_csv)
            if year_range is not None:
                df = df[
                    (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])
                ]

            total_recycle: float = df["Waste Recycle (Kg)"].sum()
            total_waste: float = df[waste_columns].sum().sum()
            recycling_rate: float = (
                total_recycle / total_waste if total_waste > 0 else 0.0
            )
            cost_delta_pct: float = round((ratio - 1.0) * 100, 4)
            cost_per_module: float = round(ratio * baseline_cost_per_kg * kg_per_w * w_per_module)

            records.append(
                {
                    "Landfill Set": label,
                    "Cost Delta (%)": cost_delta_pct,
                    "Cost per Module ($)": cost_per_module,
                    "Recycling Rate": recycling_rate,
                }
            )
            print(f"  {label} | delta={cost_delta_pct:+.1f}% | cost/module=${cost_per_module} | rate={recycling_rate:.3%}")

    if not records:
        print(f"No scenario data found in {iteration_dir}")
        return

    df_all: pd.DataFrame = pd.DataFrame(records)

    # CSV: one row per x value, one column per landfill set
    x_col: str = "Cost Delta (%)" if x_axis == "pct" else "Cost per Module ($)"
    df_pivot: pd.DataFrame = (
        df_all.pivot(
            index=x_col, columns="Landfill Set", values="Recycling Rate"
        )
        .reset_index()
        .sort_values(x_col)
    )
    df_pivot.columns.name = None
    csv_path: str = os.path.join(output_dir, f"recycling_rate_sensitivity_{sensitivity_type}.csv")
    df_pivot.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors: dict[str, str] = {
        "All Landfills": "#1f77b4",
        "True Landfills": "#ff7f0e",
    }

    for label in ["All Landfills", "True Landfills"]:
        subset: pd.DataFrame = (
            df_all[df_all["Landfill Set"] == label]
            .sort_values(x_col)
        )
        ax.plot(
            subset[x_col],
            subset["Recycling Rate"],
            marker="o",
            label=label,
            color=colors[label],
            linewidth=2,
            markersize=6,
        )

    if x_axis == "pct":
        ax.axvline(
            x=0.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline"
        )
    cost_label: str = "Recycling Cost" if sensitivity_type == "recycling" else "Transport Cost"
    if x_axis == "pct":
        ax.set_xlabel(f"{cost_label} Change (%)", fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    else:
        ax.set_xlabel(f"{cost_label} ($/module)", fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.set_ylabel("Recycling Rate", fontsize=12)
    ax.set_title(
        f"Recycling Rate Sensitivity to {cost_label}",
        fontsize=13,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path: str = os.path.join(output_dir, f"recycling_rate_sensitivity_{sensitivity_type}.jpg")
    plt.savefig(plot_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_recycling_rate_sensitivity_heatmap(
        iteration_dir: str = "results/RTN_Iteration_3",
        output_dir: str = "",
        year_range: tuple[int, int] | None = None,
    baseline_cost_per_kg: float = 0.4,
        kg_per_w: float = 0.0077,
        w_per_module: float = 270.0,
) -> None:
    """
    Plot a heatmap of simulated recycling rate as a function of recycling cost
    (x-axis, $/module) and initial recycling rate (y-axis, %).

    The directory layout expected under iteration_dir is:
        recycle_rate_<N>pct/
            RTN_run_all_landfills_<ratio>/waste_kg_per_year_by_site.csv
            RTN_run_true_landfills_<ratio>/waste_kg_per_year_by_site.csv

    Two heatmaps are saved side-by-side: All Landfills and True Landfills.
    Cost conversion uses the same formula as plot_recycling_rate_sensitivity:
        cost_per_module = round(ratio * baseline_cost_per_kg * kg_per_w * w_per_module)

    Parameters:
    iteration_dir (str): Base directory containing recycle_rate_*pct subfolders.
    output_dir (str): Directory to save outputs. Defaults to iteration_dir.
    year_range (tuple[int,int] | None): Optional (start_year, end_year) inclusive filter.
    baseline_cost_per_kg (float): Baseline recycling cost in $/kg (default 0.4).
    kg_per_w (float): Average module mass in kg/W (default 0.0077).
    w_per_module (float): Average module wattage in W (default 270).
    Returns:
    None
    """
    if not output_dir:
        output_dir = iteration_dir

    waste_columns: list[str] = [
        "Waste Repair (Kg)",
        "Waste Sell (Kg)",
        "Waste Recycle (Kg)",
        "Waste Landfill (Kg)",
        "Waste Hoard (Kg)",
    ]

    _RATE_DIR_RE = re.compile(r"recycle_rate_(\d+)pct$")
    _SET_PREFIXES: dict[str, str] = {
        "All Landfills": "RTN_run_all_landfills",
        "True Landfills": "RTN_run_true_landfills",
    }

    def _parse_cost_ratio(folder_name: str, prefix: str) -> float | None:
        suffix: str = folder_name[len(prefix):]
        if not suffix:
            return 1.0
        s: str = suffix.lstrip("_")
        if not s or s.startswith("transport"):
            return None
        try:
            return float(s[3:]) if s.startswith("neg") else float(s)
        except ValueError:
            return None

    records: list[dict] = []

    for rate_entry in sorted(os.scandir(iteration_dir), key=lambda e: e.name):
        if not rate_entry.is_dir():
            continue
        m = _RATE_DIR_RE.match(rate_entry.name)
        if not m:
            continue
        initial_rate_pct: int = int(m.group(1))

        for label, prefix in _SET_PREFIXES.items():
            seen_ratios: set[float] = set()
            for scenario_entry in sorted(os.scandir(rate_entry.path), key=lambda e: e.name):
                if not scenario_entry.is_dir() or not scenario_entry.name.startswith(prefix):
                    continue
                ratio: float | None = _parse_cost_ratio(scenario_entry.name, prefix)
                if ratio is None or ratio in seen_ratios:
                    continue
                seen_ratios.add(ratio)

                site_csv: str = os.path.join(scenario_entry.path, "waste_kg_per_year_by_site.csv")
                if not os.path.isfile(site_csv):
                    print(f"  Skipping {scenario_entry.name}: waste_kg_per_year_by_site.csv not found")
                    continue

                df: pd.DataFrame = pd.read_csv(site_csv)
                if year_range is not None:
                    df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

                rates_df: pd.DataFrame = _calculate_waste_rates(df, waste_columns)
                recycling_rate: float = float(rates_df.iloc[0]["Waste Recycle Rate"])
                cost_per_module: int = round(ratio * baseline_cost_per_kg * kg_per_w * w_per_module)

                records.append({
                    "Landfill Set": label,
                    "Initial Rate (%)": initial_rate_pct,
                    "Cost per Module ($)": cost_per_module,
                    "Recycling Rate": recycling_rate,
                })
                print(
                    f"  {label} | init={initial_rate_pct}% "
                    f"| cost/module=${cost_per_module} "
                    f"| rate={recycling_rate:.3%}"
                )

    if not records:
        print(f"No data found in {iteration_dir}")
        return

    df_all: pd.DataFrame = pd.DataFrame(records)

    # Save raw data
    csv_path: str = os.path.join(output_dir, "recycling_rate_sensitivity_heatmap.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, label in zip(axes, ["All Landfills", "True Landfills"]):
        subset: pd.DataFrame = df_all[df_all["Landfill Set"] == label]
        pivot: pd.DataFrame = subset.pivot_table(
            index="Initial Rate (%)",
            columns="Cost per Module ($)",
            values="Recycling Rate",
            aggfunc="mean",
        )
        pivot = pivot.sort_index(ascending=False)   # highest initial rate at top
        pivot = pivot[sorted(pivot.columns)]         # ascending cost left → right

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Recycling Rate", fontsize=10)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"${c}" for c in pivot.columns], fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r}%" for r in pivot.index], fontsize=10)
        ax.set_xlabel("Recycling Cost ($/module)", fontsize=11)
        ax.set_ylabel("Initial Recycling Rate (%)", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")

        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                val: float = pivot.values[row_idx, col_idx]
                if not np.isnan(val):
                    ax.text(
                        col_idx, row_idx,
                        f"{val:.1%}",
                        ha="center", va="center",
                        fontsize=9,
                        color="black" if 0.2 < val < 0.8 else "white",
                    )

    plt.suptitle(
        "Recycling Rate Sensitivity: Cost vs. Initial Recycling Rate",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out_path: str = os.path.join(output_dir, "recycling_rate_sensitivity_heatmap.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Heatmap saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot waste data by PCA components.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing the results CSV files (default: 'results')."
    )

    parser.add_argument(
        "--run_option",
        type=str,
        choices=["plot_waste_by_pca", "aggregate_consumer_results", "aggregate_tclp_results", "compare_waste_management_rates", "plot_recycling_rate_comparison", "plot_statewise_waste_management_rates", "plot_total_waste_by_state", "recycling_rate_sensitivity", "recycling_rate_sensitivity_heatmap"],
        default="plot_waste_by_pca",
        help="Choose the operation to perform (default: 'plot_waste_by_pca')."
    )

    parser.add_argument(
        "--columns",
        type=str,
        nargs='*',
        default=["Waste Repair (Kg)","Waste Sell (Kg)","Waste Recycle (Kg)","Waste Landfill (Kg)","Waste Hoard (Kg)"],
        help="List of columns to include in the aggregated consumer results (default: all columns)."
    )
    
    parser.add_argument(
        "--use_mean_method",
        action="store_true",
        help="Use mean method (calculate rates per site then average) instead of sum method (sum waste amounts then calculate rates). Default is sum method."
    )

    parser.add_argument(
        "--iteration_dir",
        type=str,
        default="results/RTN_Iteration_3",
        help="Base directory containing scenario subfolders for sensitivity analysis (default: 'results/RTN_Iteration_3')."
    )

    parser.add_argument(
        "--sensitivity_type",
        type=str,
        choices=["recycling", "transport"],
        default="recycling",
        help="Cost component for sensitivity analysis: 'recycling' or 'transport' (default: 'recycling')."
    )

    parser.add_argument(
        "--x_axis",
        type=str,
        choices=["pct", "cost_per_module"],
        default="pct",
        help="X-axis display for sensitivity plot: 'pct' for %% change (default) or 'cost_per_module' for $/module."
    )

    parser.add_argument(
        "--baseline_cost_per_kg",
        type=float,
        default=0.4,
        help="Baseline recycling cost in $/kg used for $/module conversion (default: 0.4)."
    )

    parser.add_argument(
        "--results_dir1",
        type=str,
        default="results/RTN_Iteration_3.1/att_mean_0.64/recycle_rate_20pct/RTN_run_all_landfills_30.05",
        help="Path to the first scenario directory for waste-rate comparison plots."
    )

    parser.add_argument(
        "--results_dir2",
        type=str,
        default="results/RTN_Iteration_3.1/att_mean_0.64/recycle_rate_20pct/RTN_run_true_landfills_30.05",
        help="Path to the second scenario directory for waste-rate comparison plots."
    )

    parser.add_argument(
        "--y_axis_min",
        type=float,
        default=None,
        help="Optional lower y-axis limit for recycling-rate-only comparison plots."
    )

    parser.add_argument(
        "--y_axis_max",
        type=float,
        default=None,
        help="Optional upper y-axis limit for recycling-rate-only comparison plots, for example 0.4 or 0.5."
    )

    args = parser.parse_args()

    if args.run_option == "aggregate_consumer_results":
        # Aggregate and save consumer results
        aggregate_and_save_consumer_results(results_dir=args.results_dir, columns=args.columns)
        plot_consumer_waste_by_site(results_dir=args.results_dir, value_columns=args.columns)
    elif args.run_option == "aggregate_tclp_results":
        aggregate_and_plot_consumer_tclp_results(results_dir=args.results_dir)
    elif args.run_option == "compare_waste_management_rates":
        compare_waste_management_rates(
            results_dir1=args.results_dir1,
            results_dir2=args.results_dir2,
            output_dir=args.results_dir,
            use_sum_method=not args.use_mean_method
        )
    elif args.run_option == "plot_recycling_rate_comparison":
        plot_recycling_rate_comparison(
            results_dir1=args.results_dir1,
            results_dir2=args.results_dir2,
            output_dir=args.results_dir,
            use_sum_method=not args.use_mean_method,
            y_axis_min=args.y_axis_min,
            y_axis_max=args.y_axis_max,
        )
    elif args.run_option == "plot_statewise_waste_management_rates":
        plot_statewise_waste_management_rates(results_dir=args.results_dir)
    elif args.run_option == "plot_total_waste_by_state":
        plot_total_waste_by_state(results_dir=args.results_dir)
    elif args.run_option == "recycling_rate_sensitivity":
        plot_recycling_rate_sensitivity(
            iteration_dir=args.iteration_dir,
            sensitivity_type=args.sensitivity_type,
            x_axis=args.x_axis,
            baseline_cost_per_kg=args.baseline_cost_per_kg,
        )
    elif args.run_option == "recycling_rate_sensitivity_heatmap":
        plot_recycling_rate_sensitivity_heatmap(
            iteration_dir=args.iteration_dir,
            baseline_cost_per_kg=args.baseline_cost_per_kg,
        )
    else:
        # Plot the waste data
        plot_waste_by_pca(results_dir=args.results_dir) 