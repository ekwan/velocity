"""This module tests chemistry.Species.
"""

import pytest
from velocity.chemistry import Species, Reaction, Network
import numpy as np

def test_constructor():
    """Make some simple Species objects.
    """
    species = Species("MeOH", "methanol")
    assert species.abbreviation == "MeOH"
    assert species.description == "methanol"
    assert str(species), "Species (MeOH=methanol)"

    species = Species("A")
    assert species.abbreviation == "A"
    assert species.description == "A"
    assert str(species) == "Species (A=A)"