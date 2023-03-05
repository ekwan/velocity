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

    #print()

    # check that you have to schedule at least one addition
    with pytest.raises(AssertionError, match="can't simulate if no additions have been scheduled"):
        network = Network(reactions_dict, fixed_concentrations=None)
        experiment = Experiment(network, eval_times=eval_times)
        experiment.simulate()

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
    
    # checking you have to give a volume if you want to add
    with pytest.raises(TypeError, match="required positional argument: 'volume'"):
        experiment = Experiment(network, eval_times=eval_times)
        experiment.schedule_segment(duration=200.0, concentrations=initial_concentrations_dict)

    # check eval time range
    with pytest.raises(AssertionError, match="eval time"):
        network = Network(reactions_dict, fixed_concentrations=None)
        experiment = Experiment(network, eval_times=[0,500])
        experiment.schedule_segment(duration=50.0, concentrations=initial_concentrations_dict, volume=1.0)
        experiment.schedule_segment(duration=100.0, concentrations={species_B:1.0}, volume=1.0)
        experiment.schedule_segment(duration=150.0, concentrations={species_B:1.0}, volume=1.0)
        experiment.simulate()


    # testing dilution only
    network = Network(reactions_dict, fixed_concentrations=[species_A, species_B])
    experiment = Experiment(network, eval_times=eval_times)
    experiment.schedule_segment(duration=50.0, concentrations=initial_concentrations_dict, volume=1.0)
    experiment.schedule_segment(duration=50.0, concentrations={species_B:1.0}, volume=1.0)
    experiment.schedule_segment(duration=100.0, concentrations={species_B:1.0}, volume=1.0)
    experiment.simulate()
    assert(np.allclose(experiment.df.loc[10.0],[1.0,0.0], rtol=1e-3, atol=1e-3))
    assert(np.allclose(experiment.df.loc[50.0],[0.5,0.5], rtol=1e-3, atol=1e-3))
    assert(np.allclose(experiment.df.loc[100.0],[1.0/3.0, 2.0/3.0], rtol=1e-3, atol=1e-3))
    assert(np.allclose(experiment.df.loc[200.0],[1.0/3.0, 2.0/3.0], rtol=1e-3, atol=1e-3))

    # test two additions
    network = Network(reactions_dict)
    experiment = Experiment(network, eval_times=eval_times)
    experiment.schedule_segment(duration=100.0, concentrations=initial_concentrations_dict, volume=1.0)
    experiment.schedule_segment(duration=100.0, concentrations={species_A:2.0}, volume=0.5)

    experiment.simulate()
    assert np.allclose(experiment.df.loc[100.0],
                      [np.exp(-0.05*100)*1/1.5 + 2*0.5/1.5,
                       (1.0-np.exp(-0.05*100))*1/1.5],
                       rtol=1e-3, atol=1e-3)
    assert np.allclose(experiment.df.loc[200.0],
                  [np.exp(-0.05*200)*2/3 + np.exp(-0.05*100)*2/3,
                   (1.0-np.exp(-0.05*200))*2/3 + (1.0-np.exp(-0.05*100))*2/3],
                   rtol=1e-3, atol=1e-3)

    experiment.df["mass_balance"] = experiment.df.sum(axis=1)
    first_half = experiment.df.mass_balance.loc[0.0:90.0]
    second_half = experiment.df.mass_balance.loc[100.0:200.0]
    assert np.allclose(first_half, np.ones_like(first_half))
    assert np.allclose(second_half, np.ones_like(second_half)*4.0/3.0)
    #print(experiment.df)
