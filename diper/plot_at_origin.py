"""
Implementation of the Plot_At_Origin analysis from DiPer.
Translates each trajectory to the origin and plots all trajectories together.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from diper.utils import (
    split_trajectories,
    ensure_output_dir,
    save_figure,
    save_results
)


def translate_to_origin(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Translate a trajectory to start at the origin (0,0).
    
    Parameters:
        traj: DataFrame with 'x' and 'y' columns
    
    Returns:
        DataFrame with additional 'relative_x' and 'relative_y' columns
    """
    if len(traj) == 0:
        return traj
    
    # Get the first x and y coordinates
    first_x, first_y = traj['x'].iloc[0], traj['y'].iloc[0]
    
    # Create a copy of the trajectory with additional columns
    traj = traj.copy()
    
    # Calculate relative coordinates
    traj['relative_x'] = traj['x'] - first_x
    traj['relative_y'] = traj['y'] - first_y
    
    return traj


def determine_plot_area(all_trajectories: List[pd.DataFrame]) -> Tuple[float, float, float, float]:
    """
    Determine a suitable plot area size based on trajectory dimensions.
    
    Parameters:
        all_trajectories: List of trajectory DataFrames with 'relative_x' and 'relative_y'
    
    Returns:
        Tuple of (xmin, xmax, ymin, ymax) for plotting
    """
    # Collect min and max values for all trajectories
    all_mins_x, all_maxs_x = [], []
    all_mins_y, all_maxs_y = [], []
    
    for traj in all_trajectories:
        if len(traj) > 0:
            all_mins_x.append(traj['relative_x'].min())
            all_maxs_x.append(traj['relative_x'].max())
            all_mins_y.append(traj['relative_y'].min())
            all_maxs_y.append(traj['relative_y'].max())
    
    if not all_mins_x:
        # Default if no trajectories
        return -10, 10, -10, 10
    
    # Find the overall min and max
    min_x, max_x = min(all_mins_x), max(all_maxs_x)
    min_y, max_y = min(all_mins_y), max(all_maxs_y)
    
    # Find the maximum range in either dimension
    range_x = max_x - min_x
    range_y = max_y - min_y
    L = max(range_x, range_y)
    
    # Add a margin (10% on each side)
    margin = 0.1 * L
    
    # Round to a nice number as in the original code
    L = np.ceil(1.1 * L / 10) * 10
    
    # Return plot limits
    return -L/2, L/2, -L/2, L/2


def plot_all_trajectories(all_trajectories: List[pd.DataFrame], 
                          condition_name: str,
                          output_dir: str = "output") -> plt.Figure:
    """
    Plot all trajectories starting from the origin.
    
    Parameters:
        all_trajectories: List of trajectory DataFrames with 'relative_x' and 'relative_y'
        condition_name: Name of the condition (for title and filename)
        output_dir: Directory to save output
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot each trajectory
    for i, traj in enumerate(all_trajectories):
        if len(traj) > 0:
            ax.plot(traj['relative_x'], traj['relative_y'], 
                    linewidth=1, alpha=0.7, 
                    label=f"Cell {i+1}" if i < 10 else None)
    
    # Determine plot area
    xmin, xmax, ymin, ymax = determine_plot_area(all_trajectories)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Relative X")
    ax.set_ylabel("Relative Y")
    ax.set_title(f"{condition_name} - Trajectories from Origin")
    
    # Add legend for first few trajectories
    if len(all_trajectories) > 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), 
                  fontsize='small', framealpha=0.7)
    
    # Make axes equal
    ax.set_aspect('equal')
    
    # Add origin marker
    ax.plot(0, 0, 'ko', markersize=8)
    
    return fig


def run_plot_at_origin_analysis(data: Dict[str, pd.DataFrame], 
                               output_dir: str = "output") -> Dict[str, List[pd.DataFrame]]:
    """
    Run the Plot_At_Origin analysis for all conditions.
    
    Parameters:
        data: Dictionary of DataFrames, keyed by condition name
        output_dir: Directory to save output
    
    Returns:
        Dictionary of translated trajectories lists, keyed by condition name
    """
    # Create output directory for plots
    plots_dir = ensure_output_dir(output_dir, "plots")
    
    # Store all translated trajectories for return
    all_translated_trajectories = {}
    
    # Process each condition
    for condition_name, df in data.items():
        print(f"Processing condition: {condition_name}")
        
        # Split into individual trajectories
        trajectories = split_trajectories(df)
        print(f"  Found {len(trajectories)} trajectories")
        
        # Translate each trajectory to the origin
        translated_trajectories = [translate_to_origin(traj) for traj in trajectories]
        
        # Store for return
        all_translated_trajectories[condition_name] = translated_trajectories
        
        # Plot all trajectories
        fig = plot_all_trajectories(translated_trajectories, condition_name, output_dir)
        
        # Save the figure
        save_figure(fig, plots_dir, f"plot_at_origin_{condition_name}")
        plt.close(fig)
        
        # Save the translated data
        combined_df = pd.concat(translated_trajectories, ignore_index=True)
        save_results(combined_df, output_dir, f"translated_trajectories_{condition_name}")
    
    return all_translated_trajectories


if __name__ == "__main__":
    # For testing
    from diper.utils import load_data
    
    # Load test data
    test_data = load_data("test_data.xlsx")
    
    # Run analysis
    run_plot_at_origin_analysis(test_data, "test_output")
