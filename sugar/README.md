# Data and fitting code for sugar isomerization

### Introduction

This directory contains data and analysis scripts.

### Catalyst deactivation

Catalyst deactivation data and fitting.

`catalyst_deactivation_data.xlsx` experimental datapoints for the catalyst deactivation study

`catalyst_deactivation.ipynb` analysis notebook for the catalyst deactivation study (python)

### Time course fitting

Time course fitting data, results, and analysis scripts.

`expt_data` Time course data separated into "alldata" and "trainset." The latter was used for fitting, while the former was used for validation.

`alpha_fitting.txt` output of fitting script for alpha fitting

`beta_fitting.txt` output of fitting script for beta fitting

`fit_alpha.jl` fitting script for alpha fitting (julia)

`fit_beta.jl` fitting script for beta fitting (julia)

`fit_quality.jl` script to calculate the quality of the fit (julia)

`uncertainty_alpha.jl` script to approximate the uncertainty in the alpha fit (julia)

`uncertainty_beta.jl` script to approximate the uncertainty in the beta fit (julia)

### Simulations

Stochastic simulations, data, and analysis scripts.

`asymptote-data` asymptote simulation results (conducted to steady state). Each file contains the results of 10_000 simulations. 

`transient-data` asymptote simulation results (conducted to steady state). Each file contains the results of 10_000 simulations. 

`run_asymptote.jl` script to run the asymptote simulations (julia)

`asymptote_process.jl` script to process the asymptote simulation results (julia)

`run_transient.jl` script to run the transient simulations (julia)

`transient_process.jl` script to process the transient simulation results (julia)

### Dft structures

DFT optimized structures and transition states.

`benchmarking` conformers of various structure types used to benchmark DFAs against CCSD(T) energies

`conformational search` conformers of structures/transition states conformationally sampled and optimized

`final structures` lowest energy structures from each conformational search further optimized

