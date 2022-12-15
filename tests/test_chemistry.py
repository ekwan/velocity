import unittest
from velocity.chemistry import Species, Reaction, Network

# to run tests, run from project root: python -m unittest discover -v

class TestSpecies(unittest.TestCase):
    def test_constructor(self):
        species = Species("MeOH", "methanol")
        self.assertEqual(species.abbreviation, "MeOH")
        self.assertEqual(species.description, "methanol")
        self.assertEqual(str(species), "Reagent (MeOH)")

        species = Species("A")
        self.assertEqual(species.abbreviation, "A")
        self.assertEqual(species.description, "A")
        self.assertEqual(str(species), "Reagent (A)")

class TestReaction(unittest.TestCase):
    def test_constructor_simple(self):
        species_A = Species("A")
        species_B = Species("B")

        reaction = Reaction(species_A, species_B)
        self.assertEqual(len(reaction.reactants), 1)
        self.assertEqual(reaction.reactants[species_A], 1)
        self.assertEqual(len(reaction.products), 1)
        self.assertEqual(reaction.products[species_B], 1)
        self.assertEqual(reaction.reversible, False)
        self.assertEqual(str(reaction), "A ----> B")

    def test_constructor_complex(self):
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
        self.assertEqual(len(reaction.reactants), 2)
        self.assertEqual(reaction.reactants[species_A], 2)
        self.assertEqual(reaction.reactants[species_B], 1)
        self.assertEqual(len(reaction.products), 2)
        self.assertEqual(reaction.products[species_C], 3)
        self.assertEqual(reaction.products[species_D], 4)
        self.assertEqual(reaction.reversible, True)
        self.assertEqual(str(reaction), "2 A + B <---> 3 C + 4 D")

    def test_stoichiometric_coefficient(self):
        with self.assertRaises(AssertionError):
            species_A = Species("A")
            species_B = Species("B")
            reaction = Reaction({species_A:-1}, species_B)

    def test_duplicate_species(self):
        with self.assertRaises(AssertionError):
            species_A = Species("A")
            reaction = Reaction(species_A, species_A)

class TestNetwork(unittest.TestCase):
    def test1(self):
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

if __name__ == "__main__":
    unittest.main()