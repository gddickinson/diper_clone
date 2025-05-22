# DiPer - Directional Persistence Analysis in Python

A Python implementation of the DiPer analysis toolset originally provided as Excel VBA macros in the paper:

> Gorelik, R., & Gautreau, A. (2014). Quantitative and unbiased analysis of directional persistence in cell migration. Nature Protocols, 9(8), 1931-1943.

## Overview

DiPer (Directional Persistence) is a suite of tools for analyzing cell migration trajectories, with a particular focus on directional persistence metrics. This Python implementation provides the same functionality as the original Excel macros but with the advantages of modern data analysis libraries and better reproducibility.

## Features

DiPer provides several analysis modules:

- **Plot_At_Origin**: Displays all trajectories starting from the origin
- **Make_Charts**: Creates individual charts for each trajectory
- **Sparse_Data**: Reduces data density by keeping 1 out of N frames
- **Speed**: Computes average speed for cells
- **Dir_Ratio**: Calculates directionality ratio over time
- **MSD**: Performs Mean Square Displacement analysis
- **Autocorrel**: Calculates direction autocorrelation
- **Autocorrel_NoGaps**: Direction autocorrelation with handling for stationary periods
- **Autocorrel_3D**: Direction autocorrelation for 3D trajectories
- **Vel_Cor**: Normalized velocity autocorrelation analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/diper.git

# Install dependencies
cd diper
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

DiPer can be used as a command-line tool or as a Python library.

### Command-line usage

```bash
# Run all analyses for a dataset
python -m diper.main --input trajectory_data.xlsx --output results --time-interval 1.0

# Run a specific analysis
python -m diper.main --input trajectory_data.xlsx --output results --time-interval 1.0 --analysis dir_ratio
```

### Python library usage

```python
import pandas as pd
from diper.utils import load_data
from diper.dir_ratio import run_dir_ratio_analysis

# Load your trajectory data
data = load_data("trajectory_data.xlsx")

# Run directionality ratio analysis
results = run_dir_ratio_analysis(data, time_interval=1.0, output_dir="results")
```

## Input Data Format

DiPer accepts Excel files (.xlsx, .xls) or CSV files with the following structure:

- Each worksheet in an Excel file corresponds to one experimental condition
- For CSV files, each file represents one condition
- Data should be organized with the following columns:
  - Columns 1-3: Optional metadata
  - Column 4: Frame number (must increase monotonically within each trajectory)
  - Column 5: x coordinate
  - Column 6: y coordinate
  - Column 7 (optional): z coordinate (for 3D analysis)

## Output

DiPer generates:

1. CSV/Excel files with analysis results
2. Publication-quality plots saved as PNG and PDF
3. Detailed statistics for further analysis

## Citation

If you use DiPer in your research, please cite the original paper:

```
Gorelik, R., & Gautreau, A. (2014). Quantitative and unbiased analysis of directional persistence in cell migration. Nature Protocols, 9(8), 1931-1943.
```

