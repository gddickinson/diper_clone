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

    This matches the VBA logic: a trajectory ends when the frame number
    doesn't increase (frame[r] >= frame[r+1]).

    Parameters:
        df: DataFrame with trajectory data

    Returns:
        List of DataFrames, each containing a single trajectory
    """
    if df.empty or len(df) <= 1:
        return [df] if not df.empty else []

    trajectories = []

    # Calculate frame differences to find trajectory boundaries
    frame_diffs = df['frame'].diff()

    # Find indices where frame number doesn't increase (excluding first NaN)
    # This includes both decreases and cases where frame numbers stay the same
    reset_mask = (frame_diffs <= 0) & (~frame_diffs.isna())
    traj_starts = [0] + list(df.index[reset_mask]) + [len(df)]

    # Remove duplicates and sort
    traj_starts = sorted(list(set(traj_starts)))

    # Split the dataframe at these indices
    for i in range(len(traj_starts) - 1):
        start_idx = traj_starts[i]
        end_idx = traj_starts[i + 1]

        if start_idx < end_idx:  # Only add non-empty trajectories
            traj_segment = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

            # Only include trajectories with more than one point
            if len(traj_segment) > 1:
                trajectories.append(traj_segment)

    # If no trajectories were found (shouldn't happen with valid data),
    # return the entire dataframe as one trajectory
    if not trajectories and len(df) > 1:
        trajectories = [df.copy().reset_index(drop=True)]

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
    #df.to_excel(os.path.join(output_dir, f"{filename}.xlsx"), index=False, engine='openpyxl')
