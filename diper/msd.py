"""
Implementation of the MSD (Mean Square Displacement) analysis from DiPer.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

from diper.utils import (
    split_trajectories,
    ensure_output_dir,
    save_figure,
    save_results
)


def calculate_msd(traj: pd.DataFrame, time_interval: float,
                 max_interval_fraction: float = 0.5) -> pd.DataFrame:
    """
    Calculate Mean Square Displacement for a trajectory using overlapping intervals.

    Parameters:
        traj: DataFrame with 'x' and 'y' columns
        time_interval: Time interval between frames
        max_interval_fraction: Maximum time interval as a fraction of trajectory length

    Returns:
        DataFrame with 'time_interval' and 'msd' columns
    """
    if len(traj) <= 1:
        return pd.DataFrame(columns=['time_interval', 'msd'])

    # Calculate maximum step size (as in the original code)
    max_step = int(len(traj) * max_interval_fraction)

    # Initialize results
    time_intervals = []
    msds = []

    # Loop through different time intervals
    for step in range(1, max_step + 1):
        n_intervals = len(traj) - step
        if n_intervals <= 0:
            continue

        # Calculate current time interval
        current_time_interval = step * time_interval

        # Calculate MSD for this time interval using overlapping windows
        squared_displacements = []

        for i in range(n_intervals):
            dx = traj['x'].iloc[i + step] - traj['x'].iloc[i]
            dy = traj['y'].iloc[i + step] - traj['y'].iloc[i]
            squared_displacements.append(dx**2 + dy**2)

        # Average the squared displacements
        msd = np.mean(squared_displacements)

        # Store results
        time_intervals.append(current_time_interval)
        msds.append(msd)

    # Create results DataFrame
    results = pd.DataFrame({
        'time_interval': time_intervals,
        'msd': msds
    })

    return results


def calculate_alpha_values(msd_results: pd.DataFrame,
                          fraction: float = 0.1) -> Tuple[float, float]:
    """
    Calculate alpha values from MSD results.

    Parameters:
        msd_results: DataFrame with 'time_interval' and 'avg_msd' columns
        fraction: Fraction of points to use from the low end

    Returns:
        Tuple of (alpha, r_squared)
    """
    if len(msd_results) <= 1:
        return np.nan, np.nan

    # Take only the specified fraction of points from the low end
    n_points = max(2, int(len(msd_results) * fraction))
    subset = msd_results.iloc[:n_points]

    # Calculate log values
    log_time = np.log10(subset['time_interval'])
    # Use avg_msd column, which is what's in the averaged results
    log_msd = np.log10(subset['avg_msd'])

    # Calculate line of best fit
    slope, intercept = np.polyfit(log_time, log_msd, 1)

    # Calculate R-squared
    y_pred = slope * log_time + intercept
    ss_tot = np.sum((log_msd - np.mean(log_msd))**2)
    ss_res = np.sum((log_msd - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    return slope, r_squared


def average_msd_results(all_msds: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Average MSD results from multiple trajectories.

    Parameters:
        all_msds: List of MSD result DataFrames

    Returns:
        DataFrame with averaged results
    """
    if not all_msds:
        return pd.DataFrame(columns=['time_interval', 'avg_msd', 'sem', 'n'])

    # Collect all unique time intervals
    all_time_intervals = set()
    for msd_df in all_msds:
        all_time_intervals.update(msd_df['time_interval'].values)

    all_time_intervals = sorted(all_time_intervals)

    # Initialize results
    results = {
        'time_interval': all_time_intervals,
        'msds': [[] for _ in all_time_intervals]
    }

    # Collect MSDs for each time interval
    for msd_df in all_msds:
        for i, t in enumerate(all_time_intervals):
            matching_rows = msd_df[msd_df['time_interval'] == t]
            if not matching_rows.empty:
                results['msds'][i].append(matching_rows['msd'].iloc[0])

    # Calculate averages and SEMs
    avg_msds = []
    sems = []

    for msds in results['msds']:
        if msds:
            avg = np.mean(msds)
            sem = np.std(msds) / np.sqrt(len(msds)) if len(msds) > 1 else np.nan
        else:
            avg = np.nan
            sem = np.nan

        avg_msds.append(avg)
        sems.append(sem)

    # Create results DataFrame
    df_results = pd.DataFrame({
        'time_interval': results['time_interval'],
        'avg_msd': avg_msds,
        'sem': sems,
        'n': [len(msds) for msds in results['msds']]
    })

    return df_results


def plot_msd_log_log(msd_results: Dict[str, pd.DataFrame],
                    alpha_values: Dict[str, float],
                    output_dir: str = "output") -> plt.Figure:
    """
    Plot MSD on a log-log scale for all conditions.

    Parameters:
        msd_results: Dictionary of MSD results per condition
        alpha_values: Dictionary of alpha values per condition
        output_dir: Directory to save output

    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each condition
    for condition, results in msd_results.items():
        # Skip if empty
        if results.empty:
            continue

        # Plot with error bars
        ax.errorbar(
            results['time_interval'],
            results['avg_msd'],
            yerr=results['sem'],
            label=f"{condition} (α={alpha_values[condition]:.2f})",
            marker='o',
            markersize=4,
            capsize=3
        )

    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Customize plot
    ax.set_title("Mean Square Displacement Analysis", fontsize=14)
    ax.set_ylabel("log(MSD), square units", fontsize=12)
    ax.set_xlabel("log(time interval)", fontsize=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.legend()

    return fig


def plot_msd_linear(msd_results: Dict[str, pd.DataFrame],
                   output_dir: str = "output") -> plt.Figure:
    """
    Plot MSD on a linear scale for all conditions.

    Parameters:
        msd_results: Dictionary of MSD results per condition
        output_dir: Directory to save output

    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each condition
    for condition, results in msd_results.items():
        # Skip if empty
        if results.empty:
            continue

        # Plot with error bars
        ax.errorbar(
            results['time_interval'],
            results['avg_msd'],
            yerr=results['sem'],
            label=condition,
            marker='o',
            markersize=4,
            capsize=3
        )

    # Customize plot
    ax.set_title("Mean Square Displacement Analysis (Linear Scale)", fontsize=14)
    ax.set_ylabel("MSD, square units", fontsize=12)
    ax.set_xlabel("Time interval", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    return fig


def run_msd_analysis(data: Dict[str, pd.DataFrame],
                    time_interval: float,
                    max_interval_fraction: float = 0.5,
                    alpha_fraction: float = 0.1,
                    output_dir: str = "output") -> Dict[str, Any]:
    """
    Run the MSD analysis for all conditions.

    Parameters:
        data: Dictionary of DataFrames, keyed by condition name
        time_interval: Time interval between frames
        max_interval_fraction: Maximum time interval as a fraction of trajectory length
        alpha_fraction: Fraction of points to use for alpha calculation
        output_dir: Directory to save output

    Returns:
        Dictionary of MSD results and alpha values
    """
    # Create output directory
    ensure_output_dir(output_dir)

    # Store results for each condition
    all_msd_results = {}
    all_alpha_values = {}

    # Process each condition
    for condition_name, df in data.items():
        print(f"Processing condition: {condition_name}")

        # Split into individual trajectories
        trajectories = split_trajectories(df)
        print(f"  Found {len(trajectories)} trajectories")

        # Calculate MSD for each trajectory
        msd_results = [calculate_msd(traj, time_interval, max_interval_fraction)
                      for traj in trajectories if len(traj) > 1]

        # Skip if no valid results
        if not msd_results:
            print(f"  No valid trajectories for MSD analysis")
            continue

        # Average MSD results
        avg_msd = average_msd_results(msd_results)

        # Calculate alpha value
        alpha, r_squared = calculate_alpha_values(avg_msd, alpha_fraction)

        # Store results
        all_msd_results[condition_name] = avg_msd
        all_alpha_values[condition_name] = alpha

        # Save detailed results for each trajectory
        for i, msd_df in enumerate(msd_results):
            save_results(msd_df, output_dir, f"msd_cell_{i+1}_{condition_name}")

        # Save averaged results
        save_results(avg_msd, output_dir, f"msd_avg_{condition_name}")

        # Save alpha values
        alpha_df = pd.DataFrame({
            'Condition': [condition_name],
            'Alpha': [alpha],
            'R_squared': [r_squared]
        })
        save_results(alpha_df, output_dir, f"msd_alpha_{condition_name}")

    # Save combined alpha values
    combined_alpha = pd.DataFrame({
        'Condition': list(all_alpha_values.keys()),
        'Alpha': list(all_alpha_values.values())
    })
    save_results(combined_alpha, output_dir, "msd_alpha_all")

    # Plot MSD on log-log scale
    if all_msd_results:
        fig = plot_msd_log_log(all_msd_results, all_alpha_values, output_dir)
        save_figure(fig, output_dir, "msd_log_log")
        plt.close(fig)

        # Plot MSD on linear scale
        fig = plot_msd_linear(all_msd_results, output_dir)
        save_figure(fig, output_dir, "msd_linear")
        plt.close(fig)

    return {
        'msd_results': all_msd_results,
        'alpha_values': all_alpha_values
    }


if __name__ == "__main__":
    # For testing
    from diper.utils import load_data

    # Load test data
    test_data = load_data("test_data.xlsx")

    # Run analysis
    run_msd_analysis(test_data, time_interval=1.0, output_dir="test_output")
