"""
Implementation of the Sparse_Data analysis from DiPer.
Deletes frames from migration trajectories, leaving 1 out of N frames.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from diper.utils import (
    split_trajectories,
    ensure_output_dir,
    save_results
)


def sparse_trajectory(traj: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Sparse a trajectory, keeping 1 out of N frames.
    
    Parameters:
        traj: DataFrame with trajectory data
        n: Keep 1 out of N frames
    
    Returns:
        Sparsed DataFrame
    """
    if len(traj) <= 1 or n <= 1:
        return traj
    
    # Create a copy to avoid modifying the original
    sparsed_traj = traj.iloc[::n].copy().reset_index(drop=True)
    
    return sparsed_traj


def run_sparse_data_analysis(data: Dict[str, pd.DataFrame], 
                            n: int,
                            output_dir: str = "output") -> Dict[str, pd.DataFrame]:
    """
    Run the Sparse_Data analysis for all conditions.
    
    Parameters:
        data: Dictionary of DataFrames, keyed by condition name
        n: Keep 1 out of N frames
        output_dir: Directory to save output
    
    Returns:
        Dictionary of sparsed DataFrames, keyed by condition name
    """
    if n <= 1:
        print("Warning: N must be greater than 1 for sparse_data. Returning original data.")
        return data
    
    print(f"Sparsing data, keeping 1 out of {n} frames")
    
    # Store sparsed data for return
    sparsed_data = {}
    
    # Process each condition
    for condition_name, df in data.items():
        print(f"Processing condition: {condition_name}")
        
        # Split into individual trajectories
        trajectories = split_trajectories(df)
        print(f"  Found {len(trajectories)} trajectories")
        
        # Sparse each trajectory
        sparsed_trajectories = [sparse_trajectory(traj, n) for traj in trajectories]
        
        # Combine back into a single DataFrame
        sparsed_df = pd.concat(sparsed_trajectories, ignore_index=True)
        
        # Store for return
        sparsed_data[condition_name] = sparsed_df
        
        # Save the sparsed data
        save_results(sparsed_df, output_dir, f"sparsed_{n}_{condition_name}")
    
    return sparsed_data


if __name__ == "__main__":
    # For testing
    from diper.utils import load_data
    
    # Load test data
    test_data = load_data("test_data.xlsx")
    
    # Run analysis
    sparsed_data = run_sparse_data_analysis(test_data, n=3, output_dir="test_output")
