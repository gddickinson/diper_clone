# DiPer User Manual
## Directional Persistence Analysis in Python

### Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Input Data Format](#input-data-format)
4. [Usage](#usage)
   - [Command-line Interface](#command-line-interface)
   - [Python Library Usage](#python-library-usage)
5. [Analysis Modules](#analysis-modules)
6. [Output Files](#output-files)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)
9. [Citation](#citation)

---

## Introduction

DiPer (Directional Persistence) is a Python implementation of the trajectory analysis toolset originally provided as Excel VBA macros in the paper by Gorelik & Gautreau (2014). This package provides comprehensive tools for analyzing cell migration trajectories with a focus on directional persistence metrics.

### Key Features
- **Plot_At_Origin**: Display all trajectories starting from the origin
- **Make_Charts**: Create individual charts for each trajectory
- **Sparse_Data**: Reduce data density by keeping 1 out of N frames
- **Speed**: Compute average speed for cells
- **Dir_Ratio**: Calculate directionality ratio over time
- **MSD**: Perform Mean Square Displacement analysis
- **Autocorrel**: Calculate direction autocorrelation
- **Autocorrel_NoGaps**: Direction autocorrelation with handling for stationary periods
- **Autocorrel_3D**: Direction autocorrelation for 3D trajectories
- **Vel_Cor**: Normalized velocity autocorrelation analysis

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Required packages: pandas, numpy, matplotlib, seaborn, openpyxl

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/diper.git
cd diper
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install the package:**
```bash
pip install -e .
```

### Dependencies
The following Python packages are required:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `openpyxl` - Excel file reading/writing

---

## Input Data Format

DiPer accepts Excel files (.xlsx, .xls) or CSV files with specific formatting requirements:

### Excel Files
- Each worksheet corresponds to one experimental condition
- Worksheet names become condition labels in the analysis

### CSV Files
- Each CSV file represents one experimental condition
- Filename (without extension) becomes the condition label

### Column Structure
Data should be organized with the following columns:

| Column | Index | Content | Required |
|--------|-------|---------|----------|
| 1-3 | 0-2 | Optional metadata | No |
| 4 | 3 | Frame number | Yes |
| 5 | 4 | X coordinate | Yes |
| 6 | 5 | Y coordinate | Yes |
| 7 | 6 | Z coordinate (for 3D analysis) | Optional |

### Important Notes
- **Frame numbers** must increase monotonically within each trajectory
- **New trajectories** are detected when frame numbers decrease (reset)
- **Missing values** should be handled before importing
- **Coordinates** can be in any unit (pixels, micrometers, etc.)

### Example Data Structure
```
Metadata1 | Metadata2 | Metadata3 | Frame | X    | Y    | Z
Cell1     | Exp1      | Cond1     | 0     | 10.5 | 20.3 | 5.1
Cell1     | Exp1      | Cond1     | 1     | 11.2 | 21.1 | 5.3
Cell1     | Exp1      | Cond1     | 2     | 12.0 | 22.5 | 5.8
Cell2     | Exp1      | Cond1     | 0     | 15.3 | 18.7 | 4.9
Cell2     | Exp1      | Cond1     | 1     | 16.1 | 19.2 | 5.2
```

---

## Usage

DiPer can be used in two ways: as a command-line tool or as a Python library.

### Command-line Interface

#### Basic Usage
```bash
python -m diper.main --input trajectory_data.xlsx --output results --time-interval 1.0
```

#### Command-line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | str | Required | Input file path (Excel or CSV) |
| `--output` | `-o` | str | "output" | Output directory |
| `--time-interval` | `-t` | float | 1.0 | Time interval between frames |
| `--analysis` | `-a` | str | "all" | Specific analysis to run |
| `--sparse-factor` | `-n` | int | 3 | Factor for sparse_data (keep 1 of N frames) |
| `--max-intervals` | `-m` | int | 30 | Max intervals for autocorrelation |
| `--threshold` | | float | 0.0 | Threshold for autocorrel_nogaps |
| `--plot-size` | | float | None | Plot area size for make_charts |
| `--no-plots` | | flag | False | Disable plot generation |

#### Analysis Options
- `all` - Run all analyses (default)
- `plot_at_origin` - Plot trajectories from origin
- `make_charts` - Individual trajectory charts
- `sparse_data` - Data sparsing
- `speed` - Speed analysis
- `dir_ratio` - Directionality ratio
- `msd` - Mean Square Displacement
- `autocorrel` - Direction autocorrelation
- `autocorrel_nogaps` - Autocorrelation with gap handling
- `autocorrel_3d` - 3D autocorrelation
- `vel_cor` - Velocity autocorrelation

#### Examples
```bash
# Run all analyses
python -m diper.main -i data.xlsx -o results -t 0.5

# Run only speed analysis
python -m diper.main -i data.xlsx -o results -t 1.0 -a speed

# Run MSD with custom parameters
python -m diper.main -i data.xlsx -o results -t 2.0 -a msd -m 50

# Run without generating plots
python -m diper.main -i data.xlsx -o results --no-plots
```

### Python Library Usage

#### Basic Example
```python
import pandas as pd
from diper.utils import load_data
from diper.dir_ratio import run_dir_ratio_analysis

# Load trajectory data
data = load_data("trajectory_data.xlsx")

# Run directionality ratio analysis
results = run_dir_ratio_analysis(data, time_interval=1.0, output_dir="results")
```

#### Advanced Example
```python
from diper.utils import load_data
from diper import (
    run_speed_analysis,
    run_msd_analysis,
    run_autocorrelation_analysis
)

# Load data
data = load_data("experiment.xlsx")

# Run multiple analyses
speed_results = run_speed_analysis(data, time_interval=0.5, output_dir="results")
msd_results = run_msd_analysis(data, time_interval=0.5, output_dir="results")
autocorr_results = run_autocorrelation_analysis(
    data, 
    time_interval=0.5, 
    max_intervals=40, 
    output_dir="results"
)

# Access results
for condition, result in speed_results.items():
    print(f"Condition: {condition}")
    print(f"Average speed: {result['summary']['Value'].iloc[0]:.3f}")
```

---

## Analysis Modules

### 1. Plot_At_Origin
**Purpose**: Translate all trajectories to start from the origin (0,0) and plot them together.

**Parameters**:
- Standard data input and output directory

**Output**:
- Combined plot showing all trajectories from origin
- Translated trajectory data files

**Use Case**: Visualize overall migration patterns and compare trajectory shapes across conditions.

### 2. Make_Charts
**Purpose**: Create individual plots for each trajectory with start/end markers.

**Parameters**:
- `plot_area_edge`: Fixed plot area size (optional)

**Output**:
- Individual trajectory plots organized by condition
- Separate subdirectories for each condition

**Use Case**: Quality control and detailed examination of individual cell behaviors.

### 3. Sparse_Data
**Purpose**: Reduce data density by keeping only every Nth frame.

**Parameters**:
- `n`: Keep 1 out of N frames (default: 3)

**Output**:
- Sparsed trajectory data files
- Modified dataset for subsequent analyses

**Use Case**: Reduce computational load or simulate lower temporal resolution.

### 4. Speed
**Purpose**: Calculate average instantaneous speeds for each cell and overall statistics.

**Parameters**:
- `time_interval`: Time between frames

**Output**:
- Cell-by-cell speed data
- Summary statistics (mean, SEM, count)
- Box plots with individual data points

**Calculations**:
- Instantaneous speed = distance / time_interval
- Average speed per cell = mean of instantaneous speeds

### 5. Dir_Ratio (Directionality Ratio)
**Purpose**: Calculate the ratio of straight-line distance to path length over time.

**Parameters**:
- `time_interval`: Time between frames

**Output**:
- Directionality ratio over time for each condition
- Final directionality ratios for each cell
- Time-course and endpoint plots

**Calculations**:
- Directionality ratio = d/D
- d = straight-line distance from start
- D = cumulative path length

### 6. MSD (Mean Square Displacement)
**Purpose**: Analyze displacement patterns and calculate diffusion characteristics.

**Parameters**:
- `time_interval`: Time between frames
- `max_interval_fraction`: Maximum interval as fraction of trajectory length (default: 0.5)
- `alpha_fraction`: Fraction of points for α calculation (default: 0.1)

**Output**:
- MSD vs. time interval (log-log and linear plots)
- α values (diffusion exponent) for each condition
- Individual cell MSD data

**Calculations**:
- MSD(τ) = ⟨[r(t+τ) - r(t)]²⟩
- α from slope of log(MSD) vs. log(τ)

### 7. Autocorrel (Direction Autocorrelation)
**Purpose**: Measure persistence of movement direction over time.

**Parameters**:
- `time_interval`: Time between frames
- `max_intervals`: Maximum number of intervals to calculate (default: 30)

**Output**:
- Direction autocorrelation vs. time interval
- Statistical data for each time point

**Calculations**:
- Autocorrelation = ⟨cos(θ(t+τ) - θ(t))⟩
- Based on dot product of normalized displacement vectors

### 8. Autocorrel_NoGaps
**Purpose**: Direction autocorrelation with handling for stationary periods by adding small random vectors.

**Parameters**:
- `time_interval`: Time between frames
- `max_intervals`: Maximum intervals (default: 30)
- `threshold`: Distance threshold for considering movement (default: 0.0)

**Output**:
- Direction autocorrelation accounting for gaps
- Modified trajectory data

**Use Case**: Handle trajectories with stationary periods or very small movements.

### 9. Autocorrel_3D
**Purpose**: Direction autocorrelation analysis for 3D trajectories.

**Parameters**:
- `time_interval`: Time between frames
- `max_intervals`: Maximum intervals (default: 30)

**Requirements**:
- Z coordinate column in input data

**Output**:
- 3D direction autocorrelation analysis
- Works with x, y, z coordinates

### 10. Vel_Cor (Velocity Autocorrelation)
**Purpose**: Normalized velocity autocorrelation analysis.

**Parameters**:
- `time_interval`: Time between frames
- `max_step_fraction`: Maximum step as fraction of trajectory length (default: 1/3)

**Output**:
- Velocity autocorrelation vs. time interval
- Normalized by trajectory characteristics

**Calculations**:
- Velocity autocorrelation = ⟨v(t)·v(t+τ)⟩ / ⟨v²⟩

---

## Output Files

DiPer generates comprehensive output in multiple formats:

### File Formats
- **CSV files**: Tabular data for further analysis
- **Excel files**: Same data with formatting
- **PNG files**: High-resolution plots (300 DPI)
- **PDF files**: Vector graphics for publication

### Directory Structure
```
output/
├── plots/                          # Plot_At_Origin plots
├── individual_trajectories/        # Make_Charts output
│   ├── condition1/
│   └── condition2/
├── overtime/                       # Dir_Ratio time-course data
├── lastpoint/                      # Dir_Ratio endpoint data
├── [analysis]_[condition].csv      # Analysis results
├── [analysis]_[condition].xlsx     # Analysis results
├── [analysis]_stats_[condition].csv # Detailed statistics
└── [analysis]_plot.png/pdf         # Summary plots
```

### Key Output Files

#### Speed Analysis
- `cell_speeds_[condition].csv`: Individual cell speeds
- `speed_summary_[condition].csv`: Summary statistics
- `speed_by_cell.png`: Box plot visualization

#### Dir_Ratio Analysis
- `dir_ratio_overtime_[condition].csv`: Time-course data
- `dir_ratio_last_[condition].csv`: Final ratios
- `dir_ratio_overtime.png`: Time-course plot
- `dir_ratio_lastpoint.png`: Endpoint comparison

#### MSD Analysis
- `msd_avg_[condition].csv`: Averaged MSD data
- `msd_cell_[N]_[condition].csv`: Individual cell data
- `msd_alpha_[condition].csv`: Diffusion exponents
- `msd_log_log.png`: Log-log MSD plot
- `msd_linear.png`: Linear MSD plot

#### Autocorrelation Analyses
- `autocorr_[condition].csv`: Average autocorrelation
- `autocorr_stats_[condition].csv`: Detailed statistics
- `autocorrelation.png`: Summary plot

---

## Examples

### Example 1: Basic Analysis Workflow
```bash
# Step 1: Prepare your data in Excel with proper column structure
# Step 2: Run complete analysis
python -m diper.main -i migration_data.xlsx -o results -t 0.5

# Step 3: Check results in the output directory
ls results/
```

### Example 2: Custom Analysis Pipeline
```python
from diper.utils import load_data
from diper.sparse_data import run_sparse_data_analysis
from diper.speed import run_speed_analysis
from diper.msd import run_msd_analysis

# Load data
data = load_data("experiment.xlsx")

# Apply data sparsing (keep every 5th frame)
sparse_data = run_sparse_data_analysis(data, n=5, output_dir="results")

# Analyze speed on sparsed data
speed_results = run_speed_analysis(
    sparse_data, 
    time_interval=2.5,  # Adjusted for sparsing: 0.5 * 5
    output_dir="results"
)

# Run MSD analysis
msd_results = run_msd_analysis(
    sparse_data,
    time_interval=2.5,
    max_interval_fraction=0.3,  # Use shorter intervals
    output_dir="results"
)

# Print summary
for condition in speed_results:
    print(f"Condition: {condition}")
    alpha = msd_results['alpha_values'][condition]
    print(f"  Diffusion exponent (α): {alpha:.3f}")
```

### Example 3: 3D Trajectory Analysis
```python
from diper.utils import load_data
from diper.autocorrel_3d import run_autocorrelation_3d_analysis

# Load 3D trajectory data (must have z coordinates)
data = load_data("3d_trajectories.xlsx")

# Run 3D autocorrelation analysis
results = run_autocorrelation_3d_analysis(
    data,
    time_interval=1.0,
    max_intervals=25,
    output_dir="3d_results"
)
```

### Example 4: Comparing Conditions
```python
import matplotlib.pyplot as plt
from diper.utils import load_data
from diper.dir_ratio import run_dir_ratio_analysis

# Load data with multiple conditions
data = load_data("treatment_comparison.xlsx")

# Run analysis
results = run_dir_ratio_analysis(data, time_interval=1.0, output_dir="comparison")

# Extract final directionality ratios for comparison
final_ratios = results['lastpoint']

# Print summary statistics
for condition, ratios in final_ratios.items():
    mean_ratio = np.mean(ratios)
    sem_ratio = np.std(ratios) / np.sqrt(len(ratios))
    print(f"{condition}: {mean_ratio:.3f} ± {sem_ratio:.3f} (n={len(ratios)})")
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Data Loading Errors
**Problem**: "CSV file doesn't have enough columns"
**Solution**: Ensure your data has at least 6 columns (3 metadata + frame + x + y)

**Problem**: "Sheet [name] doesn't have enough columns"
**Solution**: Check that each Excel worksheet has the required column structure

#### 2. Frame Number Issues
**Problem**: No trajectories detected or strange trajectory splitting
**Solution**: 
- Verify frame numbers increase monotonically within each trajectory
- Check that frame resets (decreases) occur only at trajectory boundaries
- Ensure frame numbers are numeric, not text

#### 3. Missing Z Coordinates
**Problem**: "Z coordinate not found" for 3D analysis
**Solution**: Add a 7th column with z coordinates, or use 2D analyses instead

#### 4. Memory Issues with Large Datasets
**Solution**: 
- Use data sparsing to reduce dataset size
- Process conditions separately
- Increase available RAM or use a more powerful computer

#### 5. Plot Generation Issues
**Problem**: Plots not appearing or saving incorrectly
**Solution**:
- Use `--no-plots` flag to skip plot generation
- Check matplotlib backend settings
- Ensure output directory is writable

#### 6. Empty Results
**Problem**: Analysis produces no output or empty results
**Solution**:
- Check that trajectories have sufficient data points (>1 or >2 depending on analysis)
- Verify time_interval parameter is appropriate for your data
- Check for NaN values in coordinate columns

### Performance Tips

1. **Data Sparsing**: Use sparse_data analysis first to reduce computational load
2. **Batch Processing**: Process conditions separately for very large datasets
3. **Parameter Tuning**: Adjust max_intervals and other parameters based on your trajectory lengths
4. **File Formats**: Use CSV for faster loading of large datasets

### Getting Help

1. **Check this manual** for parameter descriptions and examples
2. **Examine output files** to understand what each analysis produces
3. **Use smaller test datasets** to verify your workflow
4. **Check the original paper** for theoretical background on the analyses

---

## Citation

If you use DiPer in your research, please cite the original paper:

```
Gorelik, R., & Gautreau, A. (2014). Quantitative and unbiased analysis of 
directional persistence in cell migration. Nature Protocols, 9(8), 1931-1943.
```

For the Python implementation, please also acknowledge:
```
DiPer Python implementation based on original work by Roman Gorelik & Alexis Gautreau
```

---

## Version Information

- **DiPer Python Version**: 1.0.0
- **Based on**: Original Excel VBA macros from Nature Protocols (2014)
- **Python Requirements**: 3.7+
- **Last Updated**: 2024

---

*This manual covers all functionality of the DiPer Python package. For additional questions or feature requests, please refer to the project repository or contact the maintainers.*