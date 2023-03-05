"""This module tests chemistry.Experiment.
"""

import pytest
from velocity.chemistry import Species, Reaction, Network, Experiment, ExperimentBuilder
import numpy as np

@pytest.fixture
def test_system():
    """ A simple first-order system.

    A --> B

    Do not change the values here, as there are hard-coded numbers in the tests.
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
    return species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times

def test_checking_for_at_least_one_addition(test_system):
    """Ensure that check that we can't simulate without Species being added to the Experiment.
    """
    species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times = test_system
    with pytest.raises(AssertionError, match="can't simulate if no additions have been scheduled"):
        network = Network(reactions_dict, fixed_concentrations=None)
        experiment = Experiment(network, eval_times=eval_times)
        experiment.simulate()

def test_single_addition(test_system):
    """Check a simple single addition.

    Check for first-order behavior.
    """
    species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times = test_system
    network = Network(reactions_dict, fixed_concentrations=None)
    experiment = Experiment(network, eval_times=eval_times)
    experiment.schedule_segment(duration=200.0, concentrations=initial_concentrations_dict, volume=1.0)
    experiment.simulate()
    last_timepoint = experiment.df.iloc[-1].to_numpy()
    mass_balance = np.sum(last_timepoint)
    assert abs(mass_balance-1) < 0.0001
    calculated_rate_constant = -np.log(last_timepoint[0]) / 200.0
    assert abs(calculated_rate_constant-rate_constant) < 0.0001

def test_volume_is_mandatory(test_system):
    """Check specifying volume is mandatory.
    """
    species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times = test_system
    with pytest.raises(TypeError, match="required positional argument: 'volume'"):
        network = Network(reactions_dict, fixed_concentrations=None)
        experiment = Experiment(network, eval_times=eval_times)
        experiment.schedule_segment(duration=200.0, concentrations=initial_concentrations_dict)

def test_eval_times_in_range(test_system):
    """Check errors are thrown if the eval times are outside of the segments.
    """
    species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times = test_system
    with pytest.raises(AssertionError, match="eval time"):
        network = Network(reactions_dict, fixed_concentrations=None)
        experiment = Experiment(network, eval_times=[0,500])
        experiment.schedule_segment(duration=50.0, concentrations=initial_concentrations_dict, volume=1.0)
        experiment.schedule_segment(duration=100.0, concentrations={species_B:1.0}, volume=1.0)
        experiment.schedule_segment(duration=150.0, concentrations={species_B:1.0}, volume=1.0)
        experiment.simulate()

def test_dilution_only(test_system):
    """Test multiple segments with fixed concentrations to check dilution calculations.
    """
    species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times = test_system
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

def test_two_additions(test_system):
    """Test multiple segments with all concentrations allowed to vary.
    """
    species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times = test_system
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

def test_dilution_only_simple(test_system):
    """Test multiple segments with fixed concentrations to check dilution calculations.

    Use the ExperimentBuilder this time.
    """
    species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times = test_system
    network = Network(reactions_dict, fixed_concentrations=[species_A, species_B])
    builder = ExperimentBuilder(network)
    builder.schedule_start(initial_concentrations_dict, volume=1.0)
    builder.schedule_addition({species_B:1.0}, volume=1.0, time=50.0)
    builder.schedule_addition({species_B:1.0}, volume=1.0, time=100.0)
    builder.schedule_end(time=200.0)
    builder.add_eval_times(eval_times)
    experiment = builder.build()

    experiment.simulate()
    assert(np.allclose(experiment.df.loc[10.0],[1.0,0.0], rtol=1e-3, atol=1e-3))
    assert(np.allclose(experiment.df.loc[50.0],[0.5,0.5], rtol=1e-3, atol=1e-3))
    assert(np.allclose(experiment.df.loc[100.0],[1.0/3.0, 2.0/3.0], rtol=1e-3, atol=1e-3))
    assert(np.allclose(experiment.df.loc[200.0],[1.0/3.0, 2.0/3.0], rtol=1e-3, atol=1e-3))

def test_experiment_builder_full(test_system):
    """Test multiple segments with all concentrations allowed to vary.

    Use the ExperimentBuilder this time.
    """
    species_A, species_B, rate_constant, reactions_dict, initial_concentrations_dict, eval_times = test_system
    network = Network(reactions_dict)

    builder = ExperimentBuilder(network)
    with pytest.raises(AssertionError, match="must schedule start first"):
        builder.build()
    builder.schedule_start(initial_concentrations_dict, volume=1.0)
    builder.schedule_addition({species_A:2.0}, volume=0.5, time=100.0)
    with pytest.raises(AssertionError, match="must schedule end first"):
        builder.build()
    builder.schedule_end(time=200.0)
    with pytest.raises(AssertionError, match="eval time.+invalid"):
        builder.add_eval_times(300.0)
        builder.build()
    builder.eval_times = []
    builder.add_eval_times(eval_times)
    experiment = builder.build()

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