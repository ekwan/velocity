# Simulation data and analysis scripts

### Introduction

This directory contains experimental kinetic data, as well as scripts used to fit rate constants and analyze the quality of the fit.

### Experimental data

Data from the kinetic experiments. Files contain collated experiments separated by sugar type (alpha vs beta) and whether test data is excluded or included. Each file contains the following columns:

- `Run`: the run number
- `Time` (s): time in seconds
- `Sugar Abbreviation` (mol %): mol % of a given sugar

### Fitting results

`alpha_fitting.txt` and `beta_fitting.txt` contain the results of the fitting process including the training set loss function and the final parameters.

### fit_alpha/beta.jl

These scripts fit either the alpha or beta training set data and prints the results to console. This script depends on the `velocity` script in the parent directory.

In addition the following packages are required:

- `CSV`
- `DataFrames`
- `Metaheuristics`
- `Optim`

### fit_quality.jl

This file calculates the loss function against all data and prints the results to console. Additionally, graphs comparing simulated to experimental data are generated and saved to png. This script depends on the `velocity` script in the parent directory.

The following packages are required:

- `CSV`
- `DataFrames`
- `Plots`

### uncertainty_alpha/beta.jl

This file approximates the sensitivity to parameter changes. One parameter pair is systematically varied while keeping the rest constant. The resulting RMSD matrix is plotted and saved as a heatmap. This script depends on the `velocity` script in the parent directory.

In addition the following packages are required:

- `CSV`
- `DataFrames`
- `Plots`