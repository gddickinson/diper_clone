"""
DiPer: Directional Persistence Analysis in Python

A Python implementation of the DiPer analysis tools originally provided as Excel VBA macros
in the paper "Quantitative and unbiased analysis of directional persistence in cell migration"
by Roman Gorelik & Alexis Gautreau (Nature Protocols, 2014).

This package provides tools for analyzing cell migration trajectories, specifically focusing on
directional persistence metrics.

Submodules:
    - autocorrel: Direction autocorrelation (2D)
    - autocorrel_3d: Direction autocorrelation (3D)
    - autocorrel_nogaps: Direction autocorrelation without gaps
    - msd: Mean square displacement analysis
    - dir_ratio: Directionality ratio analysis
    - speed: Speed analysis
    - vel_cor: Velocity correlation analysis
    - make_charts: Chart generation utilities
    - plot_at_origin: Plot trajectories at origin
    - sparse_data: Sparse data handling
    - utils: Shared utilities (data loading, output, trajectory splitting)
"""

__version__ = "1.0.0"
__author__ = "Python implementation based on original work by Roman Gorelik & Alexis Gautreau"

__all__ = [
    "autocorrel",
    "autocorrel_3d",
    "autocorrel_nogaps",
    "msd",
    "dir_ratio",
    "speed",
    "vel_cor",
    "make_charts",
    "plot_at_origin",
    "sparse_data",
    "utils",
]

from diper.utils import load_data, ensure_output_dir
