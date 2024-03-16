# Simulation data and analysis scripts

### Introduction

This directory contains data and analysis scripts for stochastic kinetic simulations. The simulations used correlations between rate constants to investigate emergent behavior.

### Asymptote data

Asymptote data (end steady state) for the stochastic simulations. Each file contains data from 10_000 simulations, each row representing one simulation. Columns are steady state concentrations (M) for each species. 

Files are named according to the parameters used:
results_*correlation type*_*rate stdev*_*sel stdev*_*bias type*_*edge cor*.csv

### Transient data

This file contains one csv file with 10_000 rows, each row representing one simulation. Columns are titled (start_tracked) according to the start species and tracked species, where the data point is the maximum for tracked species over the course of the simulation that starts with the start species.

### run_asymptote.jl

This file runs asymptote simulations and saves the results to csv files. By default this runs 10_000 simulations. This script depends on the `velocity` script in the parent directory.

In addition the following packages are required:

- `CSV`
- `DataFrames`
- `LinearAlgebra`
- `Distributions`

### asymptote-process.jl

This file processes the asymptote simulations. This script will print the processed asymptote probabilities to the console. By default these probabilities compare to a selectivity threshold of 0.5.

The following packages are required:

- `CSV`
- `DataFrames`

### run_transient.jl

This file runs transient simulations and saves the results to csv files. By default this runs 10_000 simulations. This script depends on the `velocity` script in the parent directory.

In addition the following packages are required:

- `CSV`
- `DataFrames`
- `LinearAlgebra`
- `Distributions`

### transient-process.jl

This file processes the transient simulations. This script will print the processed transient probabilities to the console. By default these probabilities compare to a selectivity threshold of 0.5. The script will also generate a histogram graph of transient probabilities sorted by edit distance.

The following packages are required:

- `CSV`
- `DataFrames`
- `Plots`
- `Graphs`