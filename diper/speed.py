"""
Implementation of the Speed analysis from DiPer.
Computes the average speed for each cell and overall.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from diper.utils import (
    split_trajectories,
    ensure_output_dir,
    save_figure,
    save_results
)


def calculate_instantaneous_speeds(traj: pd.DataFrame, time_interval: float) -> pd.DataFrame:
    """
    Calculate instantaneous speeds for a trajectory.
    
    Parameters:
        traj: DataFrame with 'x' and 'y' columns
        time_interval: Time interval between frames
    
    Returns:
        DataFrame with additional 'distance' and 'inst_speed' columns
    """
    if len(traj) <= 1:
        return traj
    
    # Create a copy to avoid modifying the original
    traj = traj.copy()
    
    # Calculate distances between consecutive points
    dx = traj['x'].diff()
    dy = traj['y'].diff()
    traj['distance'] = np.sqrt(dx**2 + dy**2)
    
    # Calculate instantaneous speeds
    traj['inst_speed'] = traj['distance'] / time_interval
    
    return traj


def calculate_cell_average_speed(traj: pd.DataFrame) -> float:
    """
    Calculate average speed for a cell trajectory.
    
    Parameters:
        traj: DataFrame with 'inst_speed' column
    
    Returns:
        Average speed of the cell
    """
    # Skip first row as it has NaN for speed
    if len(traj) <= 1:
        return np.nan
    
    return traj['inst_speed'].iloc[1:].mean()


def plot_speed_by_cell(avg_speeds: Dict[str, List[float]], 
                      output_dir: str = "output") -> plt.Figure:
    """
    Plot average speed by cell for each condition.
    
    Parameters:
        avg_speeds: Dictionary of average speeds per condition
        output_dir: Directory to save output
    
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    conditions = []
    speeds = []
    
    for condition, speed_list in avg_speeds.items():
        for speed in speed_list:
            conditions.append(condition)
            speeds.append(speed)
    
    plot_df = pd.DataFrame({
        'Condition': conditions,
        'Average Speed': speeds
    })
    
    # Create box plot
    sns.boxplot(x='Condition', y='Average Speed', data=plot_df, ax=ax)
    
    # Add individual points
    sns.stripplot(x='Condition', y='Average Speed', data=plot_df, 
                 size=4, color='black', alpha=0.5, ax=ax)
    
    # Calculate and add mean with error bars
    summary = plot_df.groupby('Condition')['Average Speed'].agg(['mean', 'sem'])
    
    # Add mean values as text
    for i, (condition, row) in enumerate(summary.iterrows()):
        ax.text(i, row['mean'], f"{row['mean']:.2f}±{row['sem']:.2f}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Customize plot
    ax.set_title("Average Speed by Cell", fontsize=14)
    ax.set_ylabel("Speed", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    
    return fig


def run_speed_analysis(data: Dict[str, pd.DataFrame], 
                      time_interval: float,
                      output_dir: str = "output") -> Dict[str, pd.DataFrame]:
    """
    Run the Speed analysis for all conditions.
    
    Parameters:
        data: Dictionary of DataFrames, keyed by condition name
        time_interval: Time interval between frames
        output_dir: Directory to save output
    
    Returns:
        Dictionary of speed results, keyed by condition name
    """
    # Create output directory
    ensure_output_dir(output_dir)
    
    # Store results for each condition
    all_results = {}
    all_avg_speeds = {}
    
    # Process each condition
    for condition_name, df in data.items():
        print(f"Processing condition: {condition_name}")
        
        # Split into individual trajectories
        trajectories = split_trajectories(df)
        print(f"  Found {len(trajectories)} trajectories")
        
        # Calculate speeds for each trajectory
        speed_trajectories = [calculate_instantaneous_speeds(traj, time_interval) 
                             for traj in trajectories]
        
        # Calculate average speed for each cell
        cell_avg_speeds = [calculate_cell_average_speed(traj) for traj in speed_trajectories]
        cell_avg_speeds = [s for s in cell_avg_speeds if not np.isnan(s)]
        
        # Store for plotting
        all_avg_speeds[condition_name] = cell_avg_speeds
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Cell Number': range(1, len(cell_avg_speeds) + 1),
            'Average Speed': cell_avg_speeds
        })
        
        # Add summary statistics
        grand_avg = np.mean(cell_avg_speeds) if cell_avg_speeds else np.nan
        sem = np.std(cell_avg_speeds) / np.sqrt(len(cell_avg_speeds)) if len(cell_avg_speeds) > 1 else np.nan
        
        summary = pd.DataFrame({
            'Statistic': ['Grand Average', 'SEM', 'Cell Count'],
            'Value': [grand_avg, sem, len(cell_avg_speeds)]
        })
        
        # Store results
        all_results[condition_name] = {
            'cell_speeds': results,
            'summary': summary
        }
        
        # Save results
        save_results(results, output_dir, f"cell_speeds_{condition_name}")
        save_results(summary, output_dir, f"speed_summary_{condition_name}")
    
    # Create combined summary DataFrame
    combined_summary = []
    for condition, result in all_results.items():
        summary = result['summary'].copy()
        summary['Condition'] = condition
        combined_summary.append(summary)
    
    combined_summary_df = pd.concat(combined_summary, ignore_index=True)
    save_results(combined_summary_df, output_dir, "speed_summary_all")
    
    # Plot speed by cell
    if all_avg_speeds:
        fig = plot_speed_by_cell(all_avg_speeds, output_dir)
        save_figure(fig, output_dir, "speed_by_cell")
        plt.close(fig)
    
    return all_results


if __name__ == "__main__":
    # For testing
    from diper.utils import load_data
    
    # Load test data
    test_data = load_data("test_data.xlsx")
    
    # Run analysis
    run_speed_analysis(test_data, time_interval=1.0, output_dir="test_output")
