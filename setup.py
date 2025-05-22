#!/usr/bin/env python3
"""
Setup script for DiPer package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diper",
    version="1.0.0",
    author="Python implementation based on original work by Roman Gorelik & Alexis Gautreau",
    author_email="your.email@example.com",
    description="Directional Persistence Analysis for Cell Migration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/diper",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.9",
    ],
    entry_points={
        "console_scripts": [
            "diper=diper.main:main",
        ],
    },
)
