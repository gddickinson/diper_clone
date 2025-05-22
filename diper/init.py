"""
DiPer: Directional Persistence Analysis in Python

A Python implementation of the DiPer analysis tools originally provided as Excel VBA macros
in the paper "Quantitative and unbiased analysis of directional persistence in cell migration"
by Roman Gorelik & Alexis Gautreau (Nature Protocols, 2014).

This package provides tools for analyzing cell migration trajectories, specifically focusing on
directional persistence metrics.
"""

__version__ = "1.0.0"
__author__ = "Python implementation based on original work by Roman Gorelik & Alexis Gautreau"

from diper.utils import load_data, ensure_output_dir
