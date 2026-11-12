# DiPer Clone

![DiPer Trajectory Analysis](examples/diper_example.png)

A Python reimplementation of the DiPer (Directional Persistence) toolbox for analyzing cell migration trajectories. Provides direction autocorrelation, mean squared displacement, speed analysis, and other trajectory metrics.

## Features

- **Direction autocorrelation** (2D and 3D) with and without gap handling
- **Mean squared displacement (MSD)** analysis
- **Speed calculation** along trajectories
- **Directionality ratio** analysis
- **Velocity correlation**
- **Trajectory visualization** (plot at origin, sparse data handling)
- **Chart generation** from analysis results
- Modular design with shared utility functions

## Requirements

- Python 3.9+ (pyproject.toml)
- numpy, pandas, matplotlib, seaborn, openpyxl

```bash
pip install -r requirements.txt
```

## Usage

```python
from diper.utils import load_data
from diper.speed import run_speed_analysis
from diper.msd import run_msd_analysis
from diper.autocorrel import run_autocorrelation_analysis

# Excel: one sheet per condition; CSV: one file per condition.
# Columns 4, 5, 6 are frame, x, y (columns 1-3 are free metadata); a new track starts
# wherever the frame number does not increase.
data = load_data("trajectories.xlsx")
speed = run_speed_analysis(data, time_interval=1.0, output_dir="results")
msd = run_msd_analysis(data, time_interval=1.0, output_dir="results")
autocorr = run_autocorrelation_analysis(data, time_interval=1.0, max_intervals=30, output_dir="results")
```

See `docs/usage_example.py` and `docs/diper_user_manual.md` for every analysis.

## Project Structure

```
diper_clone/
├── diper/
│   ├── __init__.py
│   ├── autocorrel.py          # Direction autocorrelation (2D)
│   ├── autocorrel_3d.py       # Direction autocorrelation (3D)
│   ├── autocorrel_nogaps.py   # Autocorrelation, stops replaced by tiny random vectors
│   ├── dir_ratio.py           # Directionality ratio
│   ├── msd.py                 # Mean squared displacement
│   ├── speed.py               # Speed analysis
│   ├── vel_cor.py             # Velocity correlation
│   ├── plot_at_origin.py      # Origin-centered trajectory plots
│   ├── sparse_data.py         # Sparse data handling
│   ├── make_charts.py         # Chart generation
│   └── utils.py               # Shared utilities
├── data/                      # Your own data (git-ignored; no example data is committed)
└── docs/                      # Documentation
```

## Reference

Based on the DiPer Excel-based toolbox for cell migration analysis.


---
*Built with AI assistance from [Claude (Anthropic)](https://claude.com/).*
