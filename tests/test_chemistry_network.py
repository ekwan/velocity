"""This module tests chemistry.Network.
"""
import pytest
from velocity.chemistry import Species, Reaction, Network
import numpy as np

def test_simple_network():
    """Tests A --> B --> C.

    Ensures that the stoichiometry matrix, concentration vector, and rate vector
    calculations work.
    """
    species_A = Species("A")
    species_B = Species("B")
    species_C = Species("C")

    reaction1 = Reaction(species_A, species_B)
    reaction2 = Reaction(species_B, species_C)

    reactions_dict = {
        reaction1 : 2.0,
        reaction2 : 3.0,
    }

    network = Network(reactions_dict)

    matrix = network.stoichiometry_matrix
    expected_stoichiometry_matrix = np.array([[-1.0,  0.0],
                                              [ 1.0, -1.0],
                                              [ 0.0,  1.0]])
    assert np.allclose(network.stoichiometry_matrix, expected_stoichiometry_matrix)
    
    concentration_vector = np.array([1.0,0.1,0.0])
    rate_vector = network.get_rate_vector(concentration_vector)
    expected_rate_vector = np.array([-2.0, 1.7, 0.3])
    assert np.allclose(rate_vector, expected_rate_vector)

def test_reversible_network():
    """Tests A == B --> C.

    Ensures that the stoichiometry matrix, concentration vector, and rate vector
    calculations work.  Ensures that fixing concentrations works.
    """
    species_A = Species("A")
    species_B = Species("B")
    species_C = Species("C")

    reaction1 = Reaction(species_A, species_B, reversible=True)
    reaction2 = Reaction(species_B, species_C)

    reactions_dict = {
        reaction1 : (2.0,3.0),
        reaction2 : (4.0),
    }

    network = Network(reactions_dict, fixed_concentrations=None)

    matrix = network.stoichiometry_matrix
    expected_stoichiometry_matrix = np.array([[-1.0,  1.0,  0.0],
                                              [ 1.0, -1.0, -1.0],
                                              [ 0.0,  0.0,  1.0]])
    assert np.allclose(network.stoichiometry_matrix, expected_stoichiometry_matrix)
    
    concentration_vector = np.array([1.0,0.2,0.1])
    rate_vector = network.get_rate_vector(concentration_vector)
    expected_rate_vector = np.array([-1.4, 0.6, 0.8])
    assert np.allclose(rate_vector, expected_rate_vector)

    network2 = Network(reactions_dict, fixed_concentrations=[species_C])
    assert np.allclose(network2.stoichiometry_matrix, expected_stoichiometry_matrix)

    rate_vector = network2.get_rate_vector(concentration_vector)
    expected_rate_vector = np.array([-1.4, 0.6, 0.0])
    assert np.allclose(rate_vector, expected_rate_vector)

def test_first_order():
    """Tests A --> B is indeed first order.
    """
    species_A = Species("A")
    species_B = Species("B")
    
    reaction1 = Reaction(species_A, species_B)

    reactions_dict = {
        reaction1 : 0.05,
    }

    network = Network(reactions_dict, fixed_concentrations=None)
    t_span = (0.0, 100.0)
    t_eval = np.linspace(0,100,101)
    initial_concentrations_dict = { species_A : 1.0 }
    concentrations_df = network.simulate_timecourse(initial_concentrations_dict, t_span, t_eval)
    concentrations_df["logA"] = -np.log(concentrations_df.A)/concentrations_df.index + np.log(concentrations_df.A.iloc[0])
    expected_log = np.ones_like(concentrations_df["logA"][1:])*0.05
    assert np.allclose(concentrations_df["logA"][1:], expected_log, rtol=1e-3, atol=1e-3)

    concentrations_df["sum"] = concentrations_df.A + concentrations_df.B
    expected_sum = np.ones_like(concentrations_df["sum"])
    assert np.allclose(concentrations_df["sum"], expected_sum)

def test_catalyst():
    """Tests A + C --> B + C is indeed catalytic in C.

    Checks the rate calculations.
    """
    species_A = Species("A")
    species_B = Species("B")
    species_C = Species("C")

    reactants = { 
                    species_A : 1,
                    species_C : 1
                }

    products = {
                    species_B : 1,
                    species_C : 1
                }

    # A + C == B + C
    reaction = Reaction(reactants, products, reversible=True)

    reactions_dict = {reaction : (2.0,0.0)}
    network = Network(reactions_dict, fixed_concentrations=None)

    # check stoichiometry matrix
    expected_stoichiometry_matrix = [[-1.,  1.],
                                     [ 1., -1.],
                                     [ 0.,  0.]]
    assert np.allclose(network.stoichiometry_matrix, expected_stoichiometry_matrix)
    
    # check rate calculation
    # forward for A: -2.0*0.4*0.6=-0.48
    # forward for B: 2.0*0.4*0.6=+0.48
    concentration_vector = np.array([0.4,0.5,0.6])
    expected_rate_vector = np.array([-0.48, 0.48, 0.0])
    rate_vector = network.get_rate_vector(concentration_vector)
    assert np.allclose(rate_vector, expected_rate_vector)

    # reverse for A: 3.0*0.5*0.6=0.9
    # reverse for B: -3.0*0.5*0.6=-0.9
    reactions_dict = {reaction : (0.0,3.0)}
    network = Network(reactions_dict, fixed_concentrations=None)
    expected_rate_vector = np.array([0.9, -0.9, 0.0])
    rate_vector = network.get_rate_vector(concentration_vector)
    assert np.allclose(rate_vector, expected_rate_vector)

    # forward and reverse for A: -0.48+0.9 = 0.42
    # forward and reverse for B: 0.48-0.9 = -0.42
    reactions_dict = {reaction : (2.0,3.0)}
    network = Network(reactions_dict, fixed_concentrations=None)
    expected_rate_vector = np.array([0.42, -0.42, 0.0])
    rate_vector = network.get_rate_vector(concentration_vector)
    assert np.allclose(rate_vector, expected_rate_vector)

def test_coefficient_multiplicity():
    """Tests 2A + 3B == 4A + 2B accounts for Species on both sides correctly.
    """
    species_A = Species("A")
    species_B = Species("B")
    species_C = Species("C")

    reactants = { 
                    species_A : 2,
                    species_B : 3,
                }

    products = {
                    species_A : 4,
                    species_B : 2,
                }

    # 2A + 3B == 4A + 2B
    reaction = Reaction(reactants, products, reversible=True)

    reactions_dict = {reaction : (5.0,6.0)}
    network = Network(reactions_dict, fixed_concentrations=None)

    # check stoichiometry matrix
    expected_stoichiometry_matrix = [[  2,  -2],
                                     [ -1,  1]]
    assert np.allclose(network.stoichiometry_matrix, expected_stoichiometry_matrix)

    # check rate calculation
    # A: +2.0*5.0*0.4^2*0.5^3 - 2.0*6.0*0.4^4*0.5^2 = 0.1232
    # B:     -5.0*0.4^2*0.5^3 +     6.0*0.4^4*0.5^2 = -0.0616
    concentration_vector = np.array([0.4,0.5])
    expected_rate_vector = np.array([0.1232, -0.0616])
    rate_vector = network.get_rate_vector(concentration_vector)
    assert np.allclose(rate_vector, expected_rate_vector)