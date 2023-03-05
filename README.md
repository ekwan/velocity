# Velocity

*a Python package for running kinetic simulations*

### Introduction

*Velocity* lets you define chemical reaction networks and simulates their timecourses by numerically integrating the underlying differential equations.  Support for multiple reagent doses is provided.

### Installation

Add this folder to the PYTHONPATH.  You will need these packages:

- `numpy`
- `scipy`
- `pandas`
- `pytest`
- `pdoc`

### Quick Start

Here's how to setup a simple A --> B reaction network:

```
species_A = Species("A")
species_B = Species("B")
    
reaction1 = Reaction(species_A, species_B)

reactions_dict = {
    reaction1 : 0.05,  # rate constant
}

network = Network(reactions_dict, fixed_concentrations=None)
```

In this case, the abbreviation for `species_A` is just "A."  You can define longer names this way:

`Species("MeOH", "methanol")`

(Keep in mind abbreviations must no more than 5 characters long.)

Now, we'll tell *velocity* to simulate 100 seconds of data and that we want to start with a concentration of 1.0 for A.

```
t_span = (0.0, 100.0)
t_eval = np.linspace(0,100,101)
initial_concentrations_dict = { species_A : 1.0 }
concentrations_df = network.simulate_timecourse(initial_concentrations_dict, t_span, t_eval)
```

`concentrations_df` will contain the abbreviations for Species as the column headings and `t_eval` as the index:

```
              A         B
time
0.0    1.000000  0.000000
1.0    0.951229  0.048771
2.0    0.904837  0.095163
3.0    0.860704  0.139296
...         ...       ...
100.0  0.006756  0.993244
```

### Multiple Dosing

You can add reagents after the start of the experiment, but you'll have to specify volumes.  Using the same setup as above:

```
# your friendly helper, the ExperimentBuilder
builder = ExperimentBuilder(network)

# what concentrations to start with
builder.schedule_start(initial_concentrations_dict, volume=1.0)

# add more A at t=100
builder.schedule_addition({species_A:2.0}, volume=0.5, time=100.0)

# stop simulating here
builder.schedule_end(time=200.0)

# as before
builder.add_eval_times(eval_times)
experiment = builder.build()
experiment.simulate()
```

Here's the result in `experiment.df`:

```
             A         B
time
0     1.000000  0.000000
10    0.606492  0.393508
20    0.367789  0.632211
...
90    0.011137  0.988863
100   0.671170  0.662163
110   0.406818  0.926516
...
200   0.004539  1.328795
```

You can also specify dosing in segments of defined lengths.  See `tests/test_chemistry_experiment.py:test_two_additions()` for an example.

### Tests

To run all the tests, execute these commands in the project root:

```
export PYTHONPATH=.
pytest
```

### API Documentation

Good API-level documentation is available.  These docs can be built with [`pdoc`](https://pdoc.env).  In the project root:

`pdoc --docformat google velocity/*.py -o docs`

### Author

Eugene Kwan, 2023
