import unittest
from velocity.chemistry import Species, Reaction, Network
import numpy as np

# to run tests, run from project root: python -m unittest discover -v

class TestSpecies(unittest.TestCase):
    def test_constructor(self):
        species = Species("MeOH", "methanol")
        self.assertEqual(species.abbreviation, "MeOH")
        self.assertEqual(species.description, "methanol")
        self.assertEqual(str(species), "Species (MeOH)")

        species = Species("A")
        self.assertEqual(species.abbreviation, "A")
        self.assertEqual(species.description, "A")
        self.assertEqual(str(species), "Species (A)")

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

    # def test_duplicate_species(self):
    #     with self.assertRaises(AssertionError):
    #         species_A = Species("A")
    #         reaction = Reaction(species_A, species_A)

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

        matrix = network.stoichiometry_matrix
        expected_stoichiometry_matrix = np.array([[-1.0,  0.0],
                                                  [ 1.0, -1.0],
                                                  [ 0.0,  1.0]])
        self.assertTrue(np.allclose(network.stoichiometry_matrix, expected_stoichiometry_matrix))
        
        concentration_vector = np.array([1.0,0.1,0.0])
        rate_vector = network.get_rate_vector(concentration_vector)
        expected_rate_vector = np.array([-2.0, 1.7, 0.3])
        self.assertTrue(np.allclose(rate_vector, expected_rate_vector))

    def test2(self):
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
        self.assertTrue(np.allclose(network.stoichiometry_matrix, expected_stoichiometry_matrix))
        
        concentration_vector = np.array([1.0,0.2,0.1])
        rate_vector = network.get_rate_vector(concentration_vector)
        expected_rate_vector = np.array([-1.4, 0.6, 0.8])
        self.assertTrue(np.allclose(rate_vector, expected_rate_vector))

        network2 = Network(reactions_dict, fixed_concentrations=[species_C])
        self.assertTrue(np.allclose(network2.stoichiometry_matrix, expected_stoichiometry_matrix))
        rate_vector = network2.get_rate_vector(concentration_vector)
        expected_rate_vector = np.array([-1.4, 0.6, 0.0])
        self.assertTrue(np.allclose(rate_vector, expected_rate_vector))

    def test3(self):
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
        self.assertTrue(np.allclose(concentrations_df["logA"][1:], expected_log, rtol=1e-3, atol=1e-3))

        concentrations_df["sum"] = concentrations_df.A + concentrations_df.B
        expected_sum = np.ones_like(concentrations_df["sum"])
        self.assertTrue(np.allclose(concentrations_df["sum"], expected_sum))

    def test_catalyst(self):
        pass

    def test_coefficient_multiplicity(self):
        pass

if __name__ == "__main__":
    unittest.main()