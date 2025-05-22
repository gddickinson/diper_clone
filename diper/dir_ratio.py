"""
Implementation of the Dir_Ratio analysis from DiPer.
Plots the directionality ratio over time as well as in the last point.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any

from diper.utils import (
    split_trajectories,
    ensure_output_dir,
    save_figure,
    save_results
)


def calculate_directionality_ratio(traj: pd.DataFrame, time_interval: float) -> pd.DataFrame:
    """
    Calculate directionality ratio for a trajectory over time.
    
    Parameters:
        traj: DataFrame with 'x' and 'y' columns
        time_interval: Time interval between frames
    
    Returns:
        DataFrame with additional 'time', 'dist_to_start', 'path_length', and 'dir_ratio' columns
    """
    if len(traj) <= 1:
        return traj
    
    # Create a copy to avoid modifying the original
    traj = traj.copy()
    
    # Calculate time column
    traj['time'] = traj['frame'] * time_interval
    
    # Calculate distances between consecutive points
    dx = traj['x'].diff()
    dy = traj['y'].diff()
    traj['segment_dist'] = np.sqrt(dx**2 + dy**2)
    
    # Calculate cumulative path length (D in the paper)
    traj['path_length'] = traj['segment_dist'].cumsum()
    
    # Calculate straight-line distance from start (d in the paper)
    start_x, start_y = traj['x'].iloc[0], traj['y'].iloc[0]
    traj['dist_to_start'] = np.sqrt((traj['x'] - start_x)**2 + (traj['y'] - start_y)**2)
    
    # Calculate directionality ratio (d/D)
    traj['dir_ratio'] = traj['dist_to_start'] / traj['path_length']
    
    # Fix first row which is NaN
    traj.loc[0, 'dir_ratio'] = 1.0
    
    return traj


def align_and_average_dir_ratios(all_trajectories: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Align directionality ratios by time and calculate averages.
    
    Parameters:
        all_trajectories: List of trajectory DataFrames with 'time' and 'dir_ratio' columns
    
    Returns:
        DataFrame with time, average dir_ratio, and SEM
    """
    if not all_trajectories:
        return pd.DataFrame(columns=['time', 'avg_dir_ratio', 'sem'])
    
    # Collect all unique time points
    all_times = set()
    for traj in all_trajectories:
        if len(traj) > 0:
            all_times.update(traj['time'].values)
    
    all_times = sorted(all_times)
    
    # Initialize results
    results = {
        'time': all_times,
        'dir_ratios': [[] for _ in all_times]
    }
    
    # Collect dir_ratios for each time point
    for traj in all_trajectories:
        if len(traj) > 0:
            for i, t in enumerate(all_times):
                matching_rows = traj[traj['time'] == t]
                if not matching_rows.empty:
                    results['dir_ratios'][i].append(matching_rows['dir_ratio'].iloc[0])
    
    # Calculate averages and SEMs
    avg_dir_ratios = []
    sems = []
    
    for ratios in results['dir_ratios']:
        if ratios:
            avg = np.mean(ratios)
            sem = np.std(ratios) / np.sqrt(len(ratios)) if len(ratios) > 1 else np.nan
        else:
            avg = np.nan
            sem = np.nan
        
        avg_dir_ratios.append(avg)
        sems.append(sem)
    
    # Create results DataFrame
    df_results = pd.DataFrame({
        'time': results['time'],
        'avg_dir_ratio': avg_dir_ratios,
        'sem': sems,
        'n': [len(ratios) for ratios in results['dir_ratios']]
    })
    
    return df_results


def plot_dir_ratio_over_time(dir_ratio_results: Dict[str, pd.DataFrame],
                            output_dir: str = "output") -> plt.Figure:
    """
    Plot directionality ratio over time for all conditions.
    
    Parameters:
        dir_ratio_results: Dictionary of directionality ratio results per condition
        output_dir: Directory to save output
    
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each condition
    for condition, results in dir_ratio_results.items():
        # Skip if empty
        if results.empty:
            continue
        
        # Plot with error bars
        ax.errorbar(
            results['time'], 
            results['avg_dir_ratio'],
            yerr=results['sem'],
            label=condition,
            marker='o',
            markersize=4,
            capsize=3
        )
    
    # Customize plot
    ax.set_title("Directionality Ratio Over Time", fontsize=14)
    ax.set_ylabel("Directionality Ratio (d/D)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    return fig


def plot_dir_ratio_last_point(dir_ratio_last: Dict[str, List[float]],
                             output_dir: str = "output") -> plt.Figure:
    """
    Plot directionality ratio at the last point for all conditions.
    
    Parameters:
        dir_ratio_last: Dictionary of last directionality ratios per condition
        output_dir: Directory to save output
    
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for plotting
    conditions = []
    ratios = []
    
    for condition, ratio_list in dir_ratio_last.items():
        conditions.extend([condition] * len(ratio_list))
        ratios.extend(ratio_list)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Condition': conditions,
        'Directionality Ratio': ratios
    })
    
    # Create bar plot with error bars
    summary = plot_df.groupby('Condition')['Directionality Ratio'].agg(['mean', 'sem']).reset_index()
    
    # Plot bars
    ax.bar(
        range(len(summary)),
        summary['mean'],
        yerr=summary['sem'],
        capsize=5,
        color='skyblue',
        edgecolor='navy',
        alpha=0.7
    )
    
    # Add individual points
    for i, condition in enumerate(summary['Condition']):
        condition_data = plot_df[plot_df['Condition'] == condition]['Directionality Ratio']
        ax.scatter(
            [i] * len(condition_data),
            condition_data,
            color='navy',
            alpha=0.5,
            s=20
        )
    
    # Customize plot
    ax.set_title("Directionality Ratio at Last Point", fontsize=14)
    ax.set_ylabel("Directionality Ratio (d/D)", fontsize=12)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary['Condition'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, min(1.1, max(summary['mean']) * 1.5))
    
    # Add values on top of bars
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(
            i, 
            row['mean'] + row['sem'] + 0.02, 
            f"{row['mean']:.2f}±{row['sem']:.2f}", 
            ha='center', 
            fontweight='bold'
        )
    
    return fig


def run_dir_ratio_analysis(data: Dict[str, pd.DataFrame], 
                          time_interval: float,
                          output_dir: str = "output") -> Dict[str, Any]:
    """
    Run the Dir_Ratio analysis for all conditions.
    
    Parameters:
        data: Dictionary of DataFrames, keyed by condition name
        time_interval: Time interval between frames
        output_dir: Directory to save output
    
    Returns:
        Dictionary of directionality ratio results
    """
    # Create output directory
    overtime_dir = ensure_output_dir(output_dir, "overtime")
    lastpoint_dir = ensure_output_dir(output_dir, "lastpoint")
    
    # Store results for each condition
    dir_ratio_results = {}
    dir_ratio_last = {}
    
    # Process each condition
    for condition_name, df in data.items():
        print(f"Processing condition: {condition_name}")
        
        # Split into individual trajectories
        trajectories = split_trajectories(df)
        print(f"  Found {len(trajectories)} trajectories")
        
        # Calculate directionality ratio for each trajectory
        dir_ratio_trajectories = [calculate_directionality_ratio(traj, time_interval) 
                                for traj in trajectories]
        
        # Store last point directionality ratios
        last_ratios = []
        for traj in dir_ratio_trajectories:
            if len(traj) > 1:  # Ensure we have at least 2 points
                last_ratios.append(traj['dir_ratio'].iloc[-1])
        
        dir_ratio_last[condition_name] = last_ratios
        
        # Align and average directionality ratios
        avg_results = align_and_average_dir_ratios(dir_ratio_trajectories)
        dir_ratio_results[condition_name] = avg_results
        
        # Save results
        save_results(avg_results, overtime_dir, f"dir_ratio_overtime_{condition_name}")
        
        last_df = pd.DataFrame({
            'Cell Number': range(1, len(last_ratios) + 1),
            'Last Dir Ratio': last_ratios
        })
        save_results(last_df, lastpoint_dir, f"dir_ratio_last_{condition_name}")
    
    # Plot directionality ratio over time
    if dir_ratio_results:
        fig = plot_dir_ratio_over_time(dir_ratio_results, output_dir)
        save_figure(fig, output_dir, "dir_ratio_overtime")
        plt.close(fig)
    
    # Plot directionality ratio at last point
    if dir_ratio_last:
        fig = plot_dir_ratio_last_point(dir_ratio_last, output_dir)
        save_figure(fig, output_dir, "dir_ratio_lastpoint")
        plt.close(fig)
    
    return {
        'overtime': dir_ratio_results,
        'lastpoint': dir_ratio_last
    }


if __name__ == "__main__":
    # For testing
    from diper.utils import load_data
    
    # Load test data
    test_data = load_data("test_data.xlsx")
    
    # Run analysis
    run_dir_ratio_analysis(test_data, time_interval=1.0, output_dir="test_output")
