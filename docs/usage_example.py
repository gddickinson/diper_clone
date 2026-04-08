#!/usr/bin/env python3
"""
Example usage of the DiPer package as a Python library.
This demonstrates how to use DiPer to analyze cell migration data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Import DiPer modules
from diper.utils import load_data, ensure_output_dir
from diper.plot_at_origin import run_plot_at_origin_analysis
from diper.sparse_data import run_sparse_data_analysis
from diper.dir_ratio import run_dir_ratio_analysis
from diper.msd import run_msd_analysis
from diper.autocorrel import run_autocorrelation_analysis


def main():
    """Run a sample DiPer analysis workflow."""
    # Set parameters
    input_file = "your_data.xlsx"  # Replace with your data file
    output_dir = "diper_results"
    time_interval = 1.0  # Time interval between frames (in minutes, seconds, etc.)
    sparse_factor = 3  # Keep 1 out of 3 frames
    
    # Create output directory
    ensure_output_dir(output_dir)
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)
    print(f"Found {len(data)} condition(s): {', '.join(data.keys())}")
    
    # Example 1: Visualize all trajectories from origin
    print("\nRunning Plot_At_Origin analysis...")
    run_plot_at_origin_analysis(data, output_dir=output_dir)
    
    # Example 2: Sparse the data to reduce density
    print(f"\nRunning Sparse_Data analysis (factor: {sparse_factor})...")
    sparse_data = run_sparse_data_analysis(data, n=sparse_factor, output_dir=output_dir)
    
    # Example The remaining analyses will use the sparsed data
    print("Using sparsed data for further analyses...")
    
    # Example 3: Calculate directionality ratio
    print("\nRunning Dir_Ratio analysis...")
    dir_ratio_results = run_dir_ratio_analysis(
        sparse_data, 
        time_interval=time_interval, 
        output_dir=output_dir
    )
    
    # Example 4: Mean Square Displacement analysis
    print("\nRunning MSD analysis...")
    msd_results = run_msd_analysis(
        sparse_data, 
        time_interval=time_interval, 
        output_dir=output_dir
    )
    
    # Example 5: Direction autocorrelation analysis
    print("\nRunning Autocorrel analysis...")
    autocorr_results = run_autocorrelation_analysis(
        sparse_data, 
        time_interval=time_interval, 
        max_intervals=30, 
        output_dir=output_dir
    )
    
    # Example 6: Post-processing the results
    print("\nPost-processing results...")
    
    # Access the alpha values from MSD analysis
    alpha_values = msd_results['alpha_values']
    print("\nMSD Alpha values (slope of log-log plot):")
    for condition, alpha in alpha_values.items():
        print(f"  {condition}: {alpha:.2f}")
    
    # Get the directionality ratio at the last point for each condition
    print("\nDirectionality ratio at the last point:")
    for condition, ratios in dir_ratio_results['lastpoint'].items():
        avg_ratio = sum(ratios) / len(ratios) if ratios else float('nan')
        print(f"  {condition}: {avg_ratio:.2f} (n={len(ratios)})")
    
    print(f"\nAnalysis completed. Results saved to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
