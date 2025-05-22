"""
Implementation of the Normalized Velocity Autocorrelation analysis from DiPer.
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


def calculate_normalized_velocity_autocorrelation(traj: pd.DataFrame, 
                                                time_interval: float,
                                                max_step_fraction: float = 1/3) -> pd.DataFrame:
    """
    Calculate normalized velocity autocorrelation for a trajectory.
    
    Parameters:
        traj: DataFrame with 'x' and 'y' columns
        time_interval: Time interval between frames
        max_step_fraction: Maximum step size as a fraction of trajectory length
    
    Returns:
        DataFrame with 'time_interval' and 'vel_autocorr' columns
    """
    if len(traj) <= 2:
        return pd.DataFrame(columns=['time_interval', 'vel_autocorr'])
    
    # Calculate displacement vectors
    dx = traj['x'].diff()
    dy = traj['y'].diff()
    
    # Remove the first row (NaN due to diff)
    dx = dx.iloc[1:].reset_index(drop=True)
    dy = dy.iloc[1:].reset_index(drop=True)
    
    # Calculate normalization factor (sum of squares of displacements)
    norm = np.sum(dx**2 + dy**2) / (time_interval**2)
    
    # Maximum step size
    max_step = max(2, int(len(dx) * max_step_fraction))
    
    # Initialize results
    time_intervals = []
    vel_autocorrs = []
    
    # For each time interval
    for step in range(1, max_step + 1):
        # Actual time interval
        actual_time = step * time_interval
        
        # Calculate velocity autocorrelation
        vel_autocorr_sum = 0
        count = 0
        
        for i in range(len(dx) - step):
            # Calculate product of velocity vectors
            x_prod = dx.iloc[i] * dx.iloc[i + step]
            y_prod = dy.iloc[i] * dy.iloc[i + step]
            
            vel_autocorr_sum += (x_prod + y_prod) / (time_interval**2 * norm)
            count += 1
        
        # Skip if no valid calculations
        if count == 0:
            continue
        
        # Average the autocorrelation
        avg_vel_autocorr = vel_autocorr_sum / count
        
        # Store results
        time_intervals.append(actual_time)
        vel_autocorrs.append(avg_vel_autocorr)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'time_interval': time_intervals,
        'vel_autocorr': vel_autocorrs
    })
    
    return results


def average_vel_autocorr_results(all_vel_autocorrs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Average velocity autocorrelation results from multiple trajectories.
    
    Parameters:
        all_vel_autocorrs: List of velocity autocorrelation result DataFrames
    
    Returns:
        DataFrame with averaged results
    """
    if not all_vel_autocorrs:
        return pd.DataFrame(columns=['time_interval', 'avg_vel_autocorr', 'sem', 'n'])
    
    # Collect all unique time intervals
    all_time_intervals = set()
    for autocorr_df in all_vel_autocorrs:
        all_time_intervals.update(autocorr_df['time_interval'].values)
    
    all_time_intervals = sorted(all_time_intervals)
    
    # Initialize results
    results = {
        'time_interval': all_time_intervals,
        'vel_autocorrs': [[] for _ in all_time_intervals]
    }
    
    # Collect autocorrelations for each time interval
    for autocorr_df in all_vel_autocorrs:
        for i, t in enumerate(all_time_intervals):
            matching_rows = autocorr_df[autocorr_df['time_interval'] == t]
            if not matching_rows.empty:
                results['vel_autocorrs'][i].append(matching_rows['vel_autocorr'].iloc[0])
    
    # Calculate averages and SEMs
    avg_autocorrs = []
    sems = []
    
    for autocorrs in results['vel_autocorrs']:
        if autocorrs:
            avg = np.mean(autocorrs)
            sem = np.std(autocorrs) / np.sqrt(len(autocorrs)) if len(autocorrs) > 1 else np.nan
        else:
            avg = np.nan
            sem = np.nan
        
        avg_autocorrs.append(avg)
        sems.append(sem)
    
    # Create results DataFrame
    df_results = pd.DataFrame({
        'time_interval': results['time_interval'],
        'avg_vel_autocorr': avg_autocorrs,
        'sem': sems,
        'n': [len(autocorrs) for autocorrs in results['vel_autocorrs']]
    })
    
    return df_results


def plot_vel_autocorrelation(vel_autocorr_results: Dict[str, pd.DataFrame],
                            output_dir: str = "output") -> plt.Figure:
    """
    Plot velocity autocorrelation for all conditions.
    
    Parameters:
        vel_autocorr_results: Dictionary of velocity autocorrelation results per condition
        output_dir: Directory to save output
    
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each condition
    for condition, results in vel_autocorr_results.items():
        # Skip if empty
        if results.empty:
            continue
        
        # Plot with error bars
        ax.errorbar(
            results['time_interval'], 
            results['avg_vel_autocorr'],
            yerr=results['sem'],
            label=condition,
            marker='o',
            markersize=4,
            capsize=3
        )
    
    # Customize plot
    ax.set_title("Normalized Velocity Autocorrelation", fontsize=14)
    ax.set_ylabel("Velocity Autocorrelation", fontsize=12)
    ax.set_xlabel("Time Interval", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.2, 1.1)
    ax.legend()
    
    return fig


def run_vel_cor_analysis(data: Dict[str, pd.DataFrame], 
                        time_interval: float,
                        max_step_fraction: float = 1/3,
                        output_dir: str = "output") -> Dict[str, pd.DataFrame]:
    """
    Run the Normalized Velocity Autocorrelation analysis for all conditions.
    
    Parameters:
        data: Dictionary of DataFrames, keyed by condition name
        time_interval: Time interval between frames
        max_step_fraction: Maximum step size as a fraction of trajectory length
        output_dir: Directory to save output
    
    Returns:
        Dictionary of velocity autocorrelation results
    """
    # Create output directory
    ensure_output_dir(output_dir)
    
    # Store results for each condition
    all_vel_autocorr_results = {}
    
    # Process each condition
    for condition_name, df in data.items():
        print(f"Processing condition: {condition_name}")
        
        # Split into individual trajectories
        trajectories = split_trajectories(df)
        print(f"  Found {len(trajectories)} trajectories")
        
        # Calculate velocity autocorrelation for each trajectory
        vel_autocorr_results = [calculate_normalized_velocity_autocorrelation(
            traj, time_interval, max_step_fraction) 
            for traj in trajectories if len(traj) > 2]
        
        # Skip if no valid results
        if not vel_autocorr_results:
            print(f"  No valid trajectories for velocity autocorrelation analysis")
            continue
        
        # Average velocity autocorrelation results
        avg_vel_autocorr = average_vel_autocorr_results(vel_autocorr_results)
        
        # Store results
        all_vel_autocorr_results[condition_name] = avg_vel_autocorr
        
        # Save results
        save_results(avg_vel_autocorr, output_dir, f"vel_cor_{condition_name}")
        
        # Save detailed results for each trajectory
        stats_df = pd.DataFrame()
        for i, interval in enumerate(avg_vel_autocorr['time_interval']):
            # Collect all coefficients for this time interval
            coefs = []
            for j, autocorr_df in enumerate(vel_autocorr_results):
                matching_rows = autocorr_df[autocorr_df['time_interval'] == interval]
                if not matching_rows.empty:
                    coefs.append(matching_rows['vel_autocorr'].iloc[0])
            
            # Create DataFrame
            interval_df = pd.DataFrame({
                f"Time_{interval}": coefs + [np.nan] * (len(trajectories) - len(coefs))
            })
            
            # Concatenate
            if stats_df.empty:
                stats_df = interval_df
            else:
                stats_df = pd.concat([stats_df, interval_df], axis=1)
        
        # Save statistics
        save_results(stats_df, output_dir, f"vel_cor_stats_{condition_name}")
    
    # Plot velocity autocorrelation
    if all_vel_autocorr_results:
        fig = plot_vel_autocorrelation(all_vel_autocorr_results, output_dir)
        save_figure(fig, output_dir, "vel_cor")
        plt.close(fig)
    
    return all_vel_autocorr_results


if __name__ == "__main__":
    # For testing
    from diper.utils import load_data
    
    # Load test data
    test_data = load_data("test_data.xlsx")
    
    # Run analysis
    run_vel_cor_analysis(test_data, time_interval=1.0, output_dir="test_output")
