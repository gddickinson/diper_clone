"""
Common utilities for DiPer analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Any


def load_data(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load trajectory data from an Excel or CSV file.
    
    Parameters:
        file_path: Path to the input file (Excel or CSV)
    
    Returns:
        Dictionary with sheet names as keys and DataFrames as values
        For CSV files, the key will be the filename
    """
    if file_path.endswith(('.xlsx', '.xls')):
        # Load all sheets from Excel file
        excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        
        # Process each sheet
        processed_data = {}
        for sheet_name, df in excel_data.items():
            # Ensure the dataframe has the required columns
            if len(df.columns) < 6:
                print(f"Warning: Sheet {sheet_name} doesn't have enough columns. Skipping.")
                continue
            
            # Rename columns to standard format
            # Frame, X, Y are in columns 4, 5, 6 (index 3, 4, 5)
            columns = list(df.columns)
            df = df.copy()
            df.columns = ['col1', 'col2', 'col3', 'frame', 'x', 'y'] + columns[6:]
            
            processed_data[sheet_name] = df
        
        return processed_data
    
    elif file_path.endswith('.csv'):
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Ensure the dataframe has the required columns
        if len(df.columns) < 6:
            raise ValueError("CSV file doesn't have enough columns.")
        
        # Rename columns to standard format
        columns = list(df.columns)
        df.columns = ['col1', 'col2', 'col3', 'frame', 'x', 'y'] + columns[6:]
        
        # Use filename as the key
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return {filename: df}
    
    else:
        raise ValueError("Unsupported file format. Use Excel (.xlsx, .xls) or CSV (.csv).")


def split_trajectories(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Split a DataFrame into individual trajectories based on frame resets.
    
    Parameters:
        df: DataFrame with trajectory data
    
    Returns:
        List of DataFrames, each containing a single trajectory
    """
    trajectories = []
    
    # First identify where new trajectories start (frame number decreases)
    frame_diffs = df['frame'].diff()
    traj_starts = list(df.index[frame_diffs < 0]) + [len(df)]
    
    # If there's no frame reset, treat all as one trajectory
    if not traj_starts[:-1]:
        return [df]
    
    # Split the dataframe at these indices
    start_idx = 0
    for end_idx in traj_starts:
        trajectories.append(df.iloc[start_idx:end_idx].copy().reset_index(drop=True))
        start_idx = end_idx
    
    return trajectories


def ensure_output_dir(output_dir: str, subdir: Optional[str] = None) -> str:
    """
    Ensure the output directory exists, create it if it doesn't.
    
    Parameters:
        output_dir: Base output directory
        subdir: Optional subdirectory
    
    Returns:
        Path to the created directory
    """
    if subdir:
        dir_path = os.path.join(output_dir, subdir)
    else:
        dir_path = output_dir
        
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_figure(fig, output_dir: str, filename: str, formats: List[str] = ['png', 'pdf']):
    """
    Save a matplotlib figure in multiple formats.
    
    Parameters:
        fig: Matplotlib figure object
        output_dir: Directory to save the figure
        filename: Base filename (without extension)
        formats: List of formats to save (default: png and pdf)
    """
    for fmt in formats:
        fig.savefig(os.path.join(output_dir, f"{filename}.{fmt}"), dpi=300, bbox_inches='tight')


def save_results(df: pd.DataFrame, output_dir: str, filename: str):
    """
    Save results DataFrame to CSV and Excel.
    
    Parameters:
        df: DataFrame with results
        output_dir: Directory to save the results
        filename: Base filename (without extension)
    """
    # Save as CSV
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)
    
    # Save as Excel
    df.to_excel(os.path.join(output_dir, f"{filename}.xlsx"), index=False, engine='openpyxl')
