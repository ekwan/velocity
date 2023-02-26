"""This module tests chemistry.Reaction.
"""

import pytest
from velocity.chemistry import Species, Reaction, Network
import numpy as np

def test_constructor_simple():
    """Tests that making a simple reaction works.
    """
    species_A = Species("A")
    species_B = Species("B")

    reaction = Reaction(species_A, species_B)

    assert len(reaction.reactants) == 1
    assert reaction.reactants[species_A] == 1
    assert len(reaction.products) == 1
    assert reaction.products[species_B] == 1
    assert reaction.reversible == False
    assert str(reaction) == "A ----> B"

def test_constructor_complex():
    """Tests that making a more complex reaction gives the right stoichiometric cofficients.
    """
    species_A = Species("A")
    species_B = Species("B")
    species_C = Species("C")
    species_D = Species("D")

    reactants = { 
                    species_A : 2,
                    species_B : 1
                }

    products = {
                    species_C : 3,
                    species_D : 4
                }

    reaction = Reaction(reactants, products, reversible=True)
    assert len(reaction.reactants) == 2
    assert reaction.reactants[species_A] == 2
    assert reaction.reactants[species_B] == 1
    assert len(reaction.products) == 2
    assert reaction.products[species_C] == 3
    assert reaction.products[species_D] == 4
    assert reaction.reversible == True
    assert str(reaction) == "2 A + B <---> 3 C + 4 D"

def test_stoichiometric_coefficients():
    """Make sure you can't use invalid stoichiometric coefficients.
    """
    with pytest.raises(AssertionError):
        species_A = Species("A")
        species_B = Species("B")
        reaction = Reaction({species_A:-1}, {species_B:0})