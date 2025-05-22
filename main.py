#!/usr/bin/env python3
"""
DiPer: Directional Persistence Analysis in Python
Main entry point script.

This script provides a command-line interface to the DiPer package, allowing users
to run the different analyses on trajectory data.
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt

from diper.utils import load_data, ensure_output_dir
from diper.plot_at_origin import run_plot_at_origin_analysis
from diper.make_charts import run_make_charts_analysis
from diper.sparse_data import run_sparse_data_analysis
from diper.speed import run_speed_analysis
from diper.dir_ratio import run_dir_ratio_analysis
from diper.msd import run_msd_analysis
from diper.autocorrel import run_autocorrelation_analysis
from diper.autocorrel_nogaps import run_autocorrelation_nogaps_analysis
from diper.autocorrel_3d import run_autocorrelation_3d_analysis
from diper.vel_cor import run_vel_cor_analysis


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""
        DiPer: Directional Persistence Analysis in Python
        A suite of tools for analyzing cell migration trajectories.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Input file path (Excel or CSV)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default="output",
        help="Output directory"
    )
    
    parser.add_argument(
        "-t", "--time-interval", 
        type=float, 
        default=1.0,
        help="Time interval between frames"
    )
    
    parser.add_argument(
        "-a", "--analysis", 
        choices=[
            "all", "plot_at_origin", "make_charts", "sparse_data", 
            "speed", "dir_ratio", "msd", "autocorrel", 
            "autocorrel_nogaps", "autocorrel_3d", "vel_cor"
        ],
        default="all",
        help="Analysis to run (default: all)"
    )
    
    parser.add_argument(
        "-n", "--sparse-factor", 
        type=int, 
        default=3,
        help="Factor for sparse_data analysis (keep 1 out of N frames)"
    )
    
    parser.add_argument(
        "-m", "--max-intervals", 
        type=int, 
        default=30,
        help="Maximum number of time intervals for autocorrelation analyses"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.0,
        help="Distance threshold for autocorrel_nogaps analysis"
    )
    
    parser.add_argument(
        "--plot-size", 
        type=float, 
        default=None,
        help="Size of plot area for make_charts analysis"
    )
    
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Disable plot generation (data files only)"
    )
    
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> None:
    """Run the specified analysis on the input data."""
    start_time = time.time()
    
    # Load the data
    print(f"Loading data from {args.input}...")
    try:
        data = load_data(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print(f"Found {len(data)} condition(s): {', '.join(data.keys())}")
    
    # Make sure the output directory exists
    ensure_output_dir(args.output)
    
    # Set matplotlib to not display plots interactively
    if args.no_plots:
        plt.ioff()
    
    # Run the specified analysis
    if args.analysis == "all" or args.analysis == "plot_at_origin":
        print("\nRunning Plot_At_Origin analysis...")
        run_plot_at_origin_analysis(data, output_dir=args.output)
    
    if args.analysis == "all" or args.analysis == "make_charts":
        print("\nRunning Make_Charts analysis...")
        run_make_charts_analysis(data, plot_area_edge=args.plot_size, output_dir=args.output)
    
    if args.analysis == "all" or args.analysis == "sparse_data":
        print(f"\nRunning Sparse_Data analysis (factor: {args.sparse_factor})...")
        sparse_data = run_sparse_data_analysis(data, n=args.sparse_factor, output_dir=args.output)
        
        if args.analysis == "all":
            # Use sparsed data for subsequent analyses
            print("Using sparsed data for further analyses...")
            data = sparse_data
    
    if args.analysis == "all" or args.analysis == "speed":
        print("\nRunning Speed analysis...")
        run_speed_analysis(data, time_interval=args.time_interval, output_dir=args.output)
    
    if args.analysis == "all" or args.analysis == "dir_ratio":
        print("\nRunning Dir_Ratio analysis...")
        run_dir_ratio_analysis(data, time_interval=args.time_interval, output_dir=args.output)
    
    if args.analysis == "all" or args.analysis == "msd":
        print("\nRunning MSD analysis...")
        run_msd_analysis(data, time_interval=args.time_interval, output_dir=args.output)
    
    if args.analysis == "all" or args.analysis == "autocorrel":
        print("\nRunning Autocorrel analysis...")
        run_autocorrelation_analysis(
            data, 
            time_interval=args.time_interval, 
            max_intervals=args.max_intervals, 
            output_dir=args.output
        )
    
    if args.analysis == "all" or args.analysis == "autocorrel_nogaps":
        print(f"\nRunning Autocorrel_NoGaps analysis (threshold: {args.threshold})...")
        run_autocorrelation_nogaps_analysis(
            data, 
            time_interval=args.time_interval, 
            max_intervals=args.max_intervals, 
            threshold=args.threshold,
            output_dir=args.output
        )
    
    if args.analysis == "all" or args.analysis == "autocorrel_3d":
        print("\nRunning Autocorrel_3D analysis...")
        run_autocorrelation_3d_analysis(
            data, 
            time_interval=args.time_interval, 
            max_intervals=args.max_intervals, 
            output_dir=args.output
        )
    
    if args.analysis == "all" or args.analysis == "vel_cor":
        print("\nRunning Vel_Cor analysis...")
        run_vel_cor_analysis(data, time_interval=args.time_interval, output_dir=args.output)
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds.")
    print(f"Results saved to {os.path.abspath(args.output)}")


def main() -> None:
    """Main entry point function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Run the analysis
    run_analysis(args)


if __name__ == "__main__":
    main()
