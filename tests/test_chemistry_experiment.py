"""This module tests chemistry.Experiment.
"""

import pytest
from velocity.chemistry import Species, Reaction, Network, Experiment
import numpy as np

def test_simple_experiment():
    """Test A --> B with an extra addition step.
    """
    species_A = Species("A")
    species_B = Species("B")
    
    reaction1 = Reaction(species_A, species_B)

    reactions_dict = {
        reaction1 : 0.00001,
    }

    network = Network(reactions_dict, fixed_concentrations=None)
    initial_concentrations_dict = { species_A : 1.0 }
    
    # with only the initial addition
    print()
    initial_concentrations_dict = { species_A : 1.0 }
    eval_times = np.arange(0,210,10)
    experiment = Experiment(network, initial_concentrations_dict, max_time=200.0, eval_times=eval_times)
    experiment.simulate()
    print(experiment.df)
    print("---")

    # with two additions
    experiment = Experiment(network, initial_concentrations_dict, max_time=200.0, eval_times=eval_times, initial_volume=1.0)
    experiment.schedule_addition(when=100.0, what={ species_B : 2.0 }, volume=2.0)
    experiment.simulate()
    print(experiment.df)

    # checking you have to give a volume if you want to add
    with pytest.raises(AssertionError):
        experiment = Experiment(network, initial_concentrations_dict, max_time=200.0, eval_times=eval_times)
        experiment.schedule_addition(when=100.0, what={ species_A : 0.5 }, volume=1.0)