"""This module tests chemistry.Experiment.
"""

import pytest
from velocity.chemistry import Species, Reaction, Network, Experiment
import numpy as np

def test_simple_experiments():
    """Test A --> B with an extra addition step.
    """
    species_A = Species("A")
    species_B = Species("B")
    
    reaction1 = Reaction(species_A, species_B)

    rate_constant = 0.05
    reactions_dict = {
        reaction1 : rate_constant,
    }
    initial_concentrations_dict = { species_A : 1.0 }
    eval_times = np.arange(0,210,10)

    print()

    # with only the initial addition
    network = Network(reactions_dict, fixed_concentrations=None)
    experiment = Experiment(network, eval_times=eval_times)
    experiment.schedule_segment(duration=200.0, concentrations=initial_concentrations_dict, volume=1.0)
    experiment.simulate()
    last_timepoint = experiment.df.iloc[-1].to_numpy()
    mass_balance = np.sum(last_timepoint)
    assert abs(mass_balance-1) < 0.0001
    calculated_rate_constant = -np.log(last_timepoint[0]) / 200.0
    assert abs(calculated_rate_constant-rate_constant) < 0.0001
    
    # # checking you have to give a volume if you want to add
    # with pytest.raises(AssertionError):
    #     experiment = Experiment(network, initial_concentrations_dict, max_time=200.0, eval_times=eval_times)
    #     experiment.schedule_addition(when=100.0, what={ species_A : 0.5 }, volume=1.0)

    # # testing dilution only
    # network = Network(reactions_dict, fixed_concentrations=[species_A, species_B])
    # experiment = Experiment(network, initial_concentrations_dict, max_time=200.0, eval_times=eval_times, initial_volume=1.0)
    # experiment.schedule_addition(when=100.0, what={ species_B : 2.0 }, volume=2.0)
    # experiment.simulate()
    # for _,row in experiment.df.tail(11).iterrows():
    #     row = row.to_numpy()
    #     assert np.allclose(row, [1.0/3.0, 4.0/3.0])

    # with one addition
    # network = Network(reactions_dict, fixed_concentrations=None)
    # experiment = Experiment(network, initial_concentrations_dict, max_time=200.0, eval_times=eval_times, initial_volume=1.0)
    # experiment.schedule_addition(when=100.0, what={ species_A : 2.0 }, volume=1.0)
    # experiment.simulate()
    # last_timepoint = experiment.df.iloc[-1].to_numpy()
    # mass_balance = np.sum(last_timepoint)
    # assert abs(mass_balance-1.5) < 0.0001
    # print(experiment.df)
    #print(np.log(experiment.df))
    #print(np.exp(-0.05*100)/2.0+1)

