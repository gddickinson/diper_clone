"""
Implementation of the Direction Autocorrelation (No Gaps) analysis from DiPer.
This implementation handles gaps by adding a small random vector.
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


def normalize_vectors_nogaps(traj: pd.DataFrame, 
                            threshold: float = 0.0) -> pd.DataFrame:
    """
    Calculate normalized vectors for a trajectory, adding small random vectors for gaps.
    
    Parameters:
        traj: DataFrame with 'x' and 'y' columns
        threshold: Distance threshold below which a cell is considered idling
    
    Returns:
        DataFrame with additional 'x_vec', 'y_vec' columns (normalized vectors)
    """
    if len(traj) <= 1:
        return traj
    
    # Create a copy to avoid modifying the original
    traj = traj.copy()
    
    # Calculate displacement vectors
    dx = traj['x'].diff().shift(-1)  # Shift to get vectors in the forward direction
    dy = traj['y'].diff().shift(-1)
    
    # Calculate vector magnitudes
    magnitudes = np.sqrt(dx**2 + dy**2)
    
    # Add small random vectors for gaps (where magnitude <= threshold)
    np.random.seed(42)  # For reproducibility
    for i in range(len(traj) - 1):
        if magnitudes.iloc[i] <= threshold:
            # Generate a small random vector
            random_angle = np.random.uniform(0, 2 * np.pi)
            random_magnitude = threshold + 0.0001 + np.random.random() / 1000
            
            # Modify the original coordinates slightly
            traj.loc[traj.index[i + 1], 'x'] = traj.loc[traj.index[i], 'x'] + \
                                            random_magnitude * np.cos(random_angle)
            traj.loc[traj.index[i + 1], 'y'] = traj.loc[traj.index[i], 'y'] + \
                                            random_magnitude * np.sin(random_angle)
    
    # Recalculate displacement vectors
    dx = traj['x'].diff().shift(-1)
    dy = traj['y'].diff().shift(-1)
    magnitudes = np.sqrt(dx**2 + dy**2)
    
    # Normalize vectors
    traj['x_vec'] = dx / magnitudes
    traj['y_vec'] = dy / magnitudes
    
    # Remove the last row (which has NaN for vectors)
    traj = traj.iloc[:-1].copy()
    
    return traj


def calculate_autocorrelation(traj: pd.DataFrame, 
                             time_interval: float,
                             max_intervals: int) -> pd.DataFrame:
    """
    Calculate direction autocorrelation for a trajectory.
    
    Parameters:
        traj: DataFrame with 'x_vec' and 'y_vec' columns
        time_interval: Time interval between frames
        max_intervals: Maximum number of time intervals to calculate
    
    Returns:
        DataFrame with 'time_interval' and 'autocorrelation' columns
    """
    if len(traj) <= 1:
        return pd.DataFrame(columns=['time_interval', 'autocorrelation'])
    
    # Limit max_intervals to trajectory length
    max_intervals = min(max_intervals, len(traj) - 1)
    
    # Initialize results
    time_intervals = []
    autocorrs = []
    
    # For each time interval
    for step in range(1, max_intervals + 1):
        # Actual time interval
        actual_time = step * time_interval
        
        # Calculate autocorrelation coefficients
        autocorr_coefs = []
        
        for i in range(len(traj) - step):
            # Calculate dot product of normalized vectors
            dot_product = (traj['x_vec'].iloc[i] * traj['x_vec'].iloc[i + step] + 
                          traj['y_vec'].iloc[i] * traj['y_vec'].iloc[i + step])
            
            autocorr_coefs.append(dot_product)
        
        # Skip if no valid coefficients
        if not autocorr_coefs:
            continue
        
        # Average the coefficients
        avg_autocorr = np.mean(autocorr_coefs)
        
        # Store results
        time_intervals.append(actual_time)
        autocorrs.append(avg_autocorr)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'time_interval': time_intervals,
        'autocorrelation': autocorrs
    })
    
    return results


def average_autocorrelation_results(all_autocorrs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Average autocorrelation results from multiple trajectories.
    
    Parameters:
        all_autocorrs: List of autocorrelation result DataFrames
    
    Returns:
        DataFrame with averaged results
    """
    if not all_autocorrs:
        return pd.DataFrame(columns=['time_interval', 'avg_autocorr', 'sem', 'n'])
    
    # Collect all unique time intervals
    all_time_intervals = set()
    for autocorr_df in all_autocorrs:
        all_time_intervals.update(autocorr_df['time_interval'].values)
    
    all_time_intervals = sorted(all_time_intervals)
    
    # Initialize results
    results = {
        'time_interval': all_time_intervals,
        'autocorrs': [[] for _ in all_time_intervals]
    }
    
    # Collect autocorrelations for each time interval
    for autocorr_df in all_autocorrs:
        for i, t in enumerate(all_time_intervals):
            matching_rows = autocorr_df[autocorr_df['time_interval'] == t]
            if not matching_rows.empty:
                results['autocorrs'][i].append(matching_rows['autocorrelation'].iloc[0])
    
    # Calculate averages and SEMs
    avg_autocorrs = []
    sems = []
    
    for autocorrs in results['autocorrs']:
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
        'avg_autocorr': avg_autocorrs,
        'sem': sems,
        'n': [len(autocorrs) for autocorrs in results['autocorrs']]
    })
    
    return df_results


def plot_autocorrelation(autocorr_results: Dict[str, pd.DataFrame],
                        threshold: float,
                        output_dir: str = "output") -> plt.Figure:
    """
    Plot direction autocorrelation for all conditions.
    
    Parameters:
        autocorr_results: Dictionary of autocorrelation results per condition
        threshold: The threshold used for idling detection
        output_dir: Directory to save output
    
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each condition
    for condition, results in autocorr_results.items():
        # Skip if empty
        if results.empty:
            continue
        
        # Plot with error bars
        ax.errorbar(
            results['time_interval'], 
            results['avg_autocorr'],
            yerr=results['sem'],
            label=condition,
            marker='o',
            markersize=4,
            capsize=3
        )
    
    # Customize plot
    threshold_str = f" (Threshold = {threshold})" if threshold > 0 else ""
    ax.set_title(f"Direction Autocorrelation - No Gaps{threshold_str}", fontsize=14)
    ax.set_ylabel("Direction Autocorrelation", fontsize=12)
    ax.set_xlabel("Time Interval", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.2, 1.1)
    ax.legend()
    
    return fig


def run_autocorrelation_nogaps_analysis(data: Dict[str, pd.DataFrame], 
                                      time_interval: float,
                                      max_intervals: int = 30,
                                      threshold: float = 0.0,
                                      output_dir: str = "output") -> Dict[str, pd.DataFrame]:
    """
    Run the Direction Autocorrelation (No Gaps) analysis for all conditions.
    
    Parameters:
        data: Dictionary of DataFrames, keyed by condition name
        time_interval: Time interval between frames
        max_intervals: Maximum number of time intervals to calculate
        threshold: Distance threshold below which a cell is considered idling
        output_dir: Directory to save output
    
    Returns:
        Dictionary of autocorrelation results
    """
    # Create output directory
    ensure_output_dir(output_dir)
    
    # Store results for each condition
    all_autocorr_results = {}
    
    # Process each condition
    for condition_name, df in data.items():
        print(f"Processing condition: {condition_name}")
        
        # Split into individual trajectories
        trajectories = split_trajectories(df)
        print(f"  Found {len(trajectories)} trajectories")
        
        # Calculate normalized vectors for each trajectory (no gaps)
        vector_trajectories = [normalize_vectors_nogaps(traj, threshold) for traj in trajectories]
        
        # Calculate autocorrelation for each trajectory
        autocorr_results = [calculate_autocorrelation(traj, time_interval, max_intervals) 
                           for traj in vector_trajectories if len(traj) > 1]
        
        # Skip if no valid results
        if not autocorr_results:
            print(f"  No valid trajectories for autocorrelation analysis")
            continue
        
        # Average autocorrelation results
        avg_autocorr = average_autocorrelation_results(autocorr_results)
        
        # Store results
        all_autocorr_results[condition_name] = avg_autocorr
        
        # Save results
        threshold_str = f"_thresh_{threshold}" if threshold > 0 else ""
        save_results(avg_autocorr, output_dir, f"autocorr_nogaps{threshold_str}_{condition_name}")
        
        # Save detailed results for each trajectory
        stats_df = pd.DataFrame()
        for i, interval in enumerate(avg_autocorr['time_interval']):
            # Collect all coefficients for this time interval
            coefs = []
            for j, autocorr_df in enumerate(autocorr_results):
                matching_rows = autocorr_df[autocorr_df['time_interval'] == interval]
                if not matching_rows.empty:
                    coefs.append(matching_rows['autocorrelation'].iloc[0])
            
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
        save_results(stats_df, output_dir, f"autocorr_nogaps{threshold_str}_stats_{condition_name}")
    
    # Plot autocorrelation
    if all_autocorr_results:
        fig = plot_autocorrelation(all_autocorr_results, threshold, output_dir)
        threshold_str = f"_thresh_{threshold}" if threshold > 0 else ""
        save_figure(fig, output_dir, f"autocorrelation_nogaps{threshold_str}")
        plt.close(fig)
    
    return all_autocorr_results


if __name__ == "__main__":
    # For testing
    from diper.utils import load_data
    
    # Load test data
    test_data = load_data("test_data.xlsx")
    
    # Run analysis
    run_autocorrelation_nogaps_analysis(
        test_data, time_interval=1.0, max_intervals=30, 
        threshold=0.2, output_dir="test_output"
    )
