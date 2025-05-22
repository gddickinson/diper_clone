"""
Implementation of the Make_Charts analysis from DiPer.
Plots each trajectory individually next to corresponding data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple

from diper.utils import (
    split_trajectories,
    ensure_output_dir,
    save_figure
)


def determine_plot_size(traj: pd.DataFrame, 
                       plot_area_edge: Optional[float] = None) -> Tuple[float, float, float, float]:
    """
    Determine plot area size for a single trajectory.
    
    Parameters:
        traj: DataFrame with 'x' and 'y' columns
        plot_area_edge: User-specified plot area size (optional)
    
    Returns:
        Tuple of (xmin, xmax, ymin, ymax) for plotting
    """
    if len(traj) == 0:
        return -10, 10, -10, 10
    
    min_x, max_x = traj['x'].min(), traj['x'].max()
    min_y, max_y = traj['y'].min(), traj['y'].max()
    
    # If user specified a plot area size, use that
    if plot_area_edge is not None:
        # Use the center of the trajectory as the midpoint
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        
        # Set limits around midpoint
        half_edge = plot_area_edge / 2
        return mid_x - half_edge, mid_x + half_edge, mid_y - half_edge, mid_y + half_edge
    
    # Otherwise, calculate based on trajectory size
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Use the larger range with a margin
    max_range = max(x_range, y_range)
    margin = 0.1 * max_range  # 10% margin
    
    # If range is very small, use a minimum size
    if max_range < 1:
        max_range = 1
        margin = 0.1
    
    return (min_x - margin, max_x + margin, 
            min_y - margin, max_y + margin)


def plot_single_trajectory(traj: pd.DataFrame, 
                          cell_number: int, 
                          condition_name: str,
                          plot_area_edge: Optional[float] = None) -> plt.Figure:
    """
    Plot a single trajectory.
    
    Parameters:
        traj: DataFrame with 'x' and 'y' columns
        cell_number: Cell/trajectory number
        condition_name: Name of the condition
        plot_area_edge: User-specified plot area size (optional)
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the trajectory
    ax.plot(traj['x'], traj['y'], 'b-', linewidth=1.5)
    
    # Mark start and end points
    if len(traj) > 0:
        ax.plot(traj['x'].iloc[0], traj['y'].iloc[0], 'go', markersize=8, label='Start')
        ax.plot(traj['x'].iloc[-1], traj['y'].iloc[-1], 'ro', markersize=8, label='End')
    
    # Determine plot area
    xmin, xmax, ymin, ymax = determine_plot_size(traj, plot_area_edge)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{condition_name}, Cell No. {cell_number}")
    
    # Add legend
    ax.legend()
    
    # Make axes equal
    ax.set_aspect('equal')
    
    return fig


def run_make_charts_analysis(data: Dict[str, pd.DataFrame], 
                           plot_area_edge: Optional[float] = None,
                           output_dir: str = "output") -> None:
    """
    Run the Make_Charts analysis for all conditions.
    
    Parameters:
        data: Dictionary of DataFrames, keyed by condition name
        plot_area_edge: User-specified plot area size (optional)
        output_dir: Directory to save output
    """
    # Create output directory for individual trajectory plots
    plots_dir = ensure_output_dir(output_dir, "individual_trajectories")
    
    # Process each condition
    for condition_name, df in data.items():
        print(f"Processing condition: {condition_name}")
        
        # Split into individual trajectories
        trajectories = split_trajectories(df)
        print(f"  Found {len(trajectories)} trajectories")
        
        # Create a subdirectory for this condition
        condition_dir = ensure_output_dir(plots_dir, condition_name)
        
        # Plot each trajectory
        for i, traj in enumerate(trajectories):
            cell_number = i + 1
            
            # Skip empty trajectories
            if len(traj) <= 1:
                print(f"  Skipping trajectory {cell_number} (insufficient data points)")
                continue
                
            # Plot the trajectory
            fig = plot_single_trajectory(traj, cell_number, condition_name, plot_area_edge)
            
            # Save the figure
            save_figure(fig, condition_dir, f"cell_{cell_number}")
            plt.close(fig)
    
    print(f"Individual trajectory plots saved to: {plots_dir}")


if __name__ == "__main__":
    # For testing
    from diper.utils import load_data
    
    # Load test data
    test_data = load_data("test_data.xlsx")
    
    # Run analysis
    run_make_charts_analysis(test_data, plot_area_edge=50, output_dir="test_output")
