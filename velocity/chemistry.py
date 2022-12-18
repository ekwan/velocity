import numpy as np

class Species():
    """ This represents a single chemical species.

        Attributes:
            abbreviation (str): abbreviation
            description (str): a longer name
    """
    def __init__(self,
                 abbreviation,
                 description=None):
        """Constructor.

        Args:
            abbreviation (str): a short moniker
            description (str, optional): a longer description
        """

        # sanity checks
        assert isinstance(abbreviation, str), f"got type {type(abbreviation)} but expected str"
        abbreviation = abbreviation.strip()
        assert 1 <= len(abbreviation) <= 5, f"got {len(abbreviation)} characters, but must be between 1-5"
        self.abbreviation = abbreviation

        if description is None:
            description = abbreviation
        assert isinstance(description, str), f"got type {type(description)}, but expected str"
        description = description.strip()
        assert len(description) > 0, "description cannot be blank"
        self.description = description

    def __str__(self):
        return f"Reagent ({self.abbreviation})"

    def __repr__(self):
        return str(self)

class Reaction():
    """ This represents a chemical reaction. 

        Attributes:
            reactants (dict): { Species : stoichiometric coefficient (int) }
            products (dict): { Species : stoichiometric coefficient (int) }
            reversible (bool): whether this reaction is reversible
    """
    def __init__(self, reactants, products, reversible=False):
        """Constructs a Reaction.

        The same species should not appear on both sides of the reaction.

        Args:
            reactants (`Species` or `dict`, {Species : stoichiometric coefficient (int)}): 
                `dict` containing `Species` as keys and their stoichiometric coefficients as values.
                If a lone Species is provided, it is assumed that its stoichiometric coefficient is 1.
            products (`dict`, {Species : stoichiometric coefficient}): same format as reactants
            reversible (bool, optional): whether this reaction can go backwards
                                         as well as forwards (default=False)
        """

        # allow reactants/products to be lone Species
        if isinstance(reactants, Species):
            reactants = { reactants : 1 }
        if isinstance(products, Species):
            products = { products : 1 }

        # read species and stoichiometric coefficients
        for d in [reactants, products]:
            assert isinstance(d, dict), f"got type {type(l)} but expected dict"
            assert len(d) > 0, "must have at least one species"
            for s,c in d.items():
                assert isinstance(s, Species), f"got type {type(s)} for {s} but expected Species"
                assert isinstance(c, int), f"got type {type(c)} for {c} but expected int"
                assert c > 0, f"stoichiometric coefficients must be > 0"
        self.reactants = reactants
        self.products = products

        # check species are not on both sides
        for s in reactants:
            assert s not in products, f"{s} is in both the reactants and products"

        # set reversibility flag
        assert isinstance(reversible, bool), f"got type {type(reversible)} but expected bool"
        self.reversible = reversible

    def __str__(self):
        def describe(d, description=None):
            if not description:
                description = ""
            for i,(s,c) in enumerate(d.items()):
                if c > 1:
                    description += f"{c} "
                description += f"{s.abbreviation}"
                if i < len(self.reactants) - 1:
                    description += " + "
            return description
        description = describe(self.reactants)
        description += " <---> " if self.reversible else " ----> "
        description = describe(self.products, description)
        return description

class Network():
    """ This represents a chemical reaction network.

        Attributes:
            species (:obj:`list`): list of Species in alphabetical order by description with no duplications
            reactions_dict (:obj:`dict`): {Reaction : rate_constant (k_forward (float), or (k_forward, k_reverse))}
            rate_constant_vector (:obj:`list`): list of all the rate constants, order matches reactions_dict
            rate_indices (:obj:`list`): list of lists, outer index corresponds to reaction, inner lists contain
                                        the indices of the Species that are on the left hand side of the reaction
            fixed_concentrations_vector (:obj:`np.ndarray` of float): indexed by order in species, elements are 0 if the
                                                                      corresponding Species has a fixed concentration or 1 otherwise
            stoichiometry_matrix (:obj:`np.ndarray` of float): rows are species, indexed by order in `species`;
                                                               columns refer to reactions, indexed by the order in `reactions_dict`
                                                               (remember, reversible reactions count as two reactions with the
                                                               forward reaction first)

    """
    def __init__(self, reactions_dict, fixed_concentrations=None):
        """Constructs a chemical reaction network.

        Args:
            reactions_dict (dict, `Reaction` : float): a dictionary that maps Reactions to their rate constants
                                                       (a single float for irreversible reactions and a tuple of two floats
                                                        for reversible reactions corresponding to the forward and reverse
                                                        rate constants in that order)
            fixed_concentrations (None or :obj:`list` of Species): these concentrations will not be updated (default: None)
        """
        # initialize fields
        species = set()
        rate_constants = []

        # sanity checks
        assert isinstance(reactions_dict, dict)
        assert len(reactions_dict) > 0, "can't have an empty reaction network"

        # rate constants must be positive floats
        def is_valid_rate_constant(item, reaction):
            assert isinstance(v, float), f"expected float for rate constant but got {type(v)} for {reaction}"
            assert v > 0.0, f"rate constant invalid for {reaction}"
            return True

        # check that all reactions and rate constants are valid
        for reaction,v in reactions_dict.items():
            assert isinstance(reaction, Reaction), f"expected Reaction but got {type(reaction)}"
            if reaction.reversible:
                assert isinstance(v, tuple), f"expected tuple for reversible rate constants but got {type(v)} for {reaction}"
                assert len(v) == 2, f"expected 2-tuple but got length {len(v)} for {reaction}"
                for rate_constant in v:
                    assert is_valid_rate_constant(rate_constant, reaction)
                rate_constants.extend(v)
            else:
                rate_constant = v
                assert is_valid_rate_constant(rate_constant, reaction)
                rate_constants.append(rate_constant)

            # store all the unique Species in a set
            species.update(reaction.reactants)
            species.update(reaction.products)

        # keep a list of all the unique Species in alphabetical order 
        species = list(sorted(species, key=lambda s : s.description))

        # check fixed concentrations
        if fixed_concentrations is None:
            fixed_concentrations = []
        for s in fixed_concentrations:
            assert isinstance(s, Species), f"expected Species but got {type(s)}"
            assert s in species, f"Species {s} is not in this reaction network"
        fixed_concentrations_vector = [0 if s in fixed_concentrations else 1 for s in species]

        # create an array of which species concentrations should be multiplied together
        # to generate the rate vector quickly
        rate_indices = []
        for reaction in reactions_dict:
            indices = [ species.index(reactant) for reactant in reaction.reactants ]
            rate_indices.append(indices)
            if reaction.reversible:
                indices = [ species.index(product) for product in reaction.product ]
                rate_indices.append(indices)

        # store data
        self.species = species
        self.reactions_dict = reactions_dict
        self.rate_constant_vector = np.array(rate_constants)
        self.rate_indices = rate_indices
        self.fixed_concentrations_vector = np.array(fixed_concentrations_vector)
        self.stoichiometry_matrix = Network._make_stoichiometry_matrix(species, reactions_dict)

    @staticmethod
    def _make_stoichiometry_matrix(all_species, reactions_dict):
        """Compute rate constant matrix from stoichiometric coefficients

        Args:
            all_species (:obj:`list` of Species): all the Species in the system
            reactions_dict (:obj:`dict`): {Reaction : rate_constant (k_forward (float), or (k_forward, k_reverse))}

        Returns:
            np.ndarray: (n_species,n_reactions) with +1 for creation and -1 for destruction
        """
        # form stoichiometry matrix
        n_reactions = 0
        for reaction in reactions_dict:
            n_reactions += 2 if reaction.reversible else 1
        n_species = len(all_species)
        matrix = np.zeros((n_species,n_reactions))

        # update the stoichiometry matrix for a given reaction
        #
        # sign refers to whether the rate constant is increasing or decreasing the amount
        # of this Species as time goes forward.  should be -1 for reactants and +1 for products
        def update_matrix(species, reaction_index, sign):
            for s in species:
                species_index = all_species.index(s)
                matrix[species_index][reaction_index] = sign

        # populate stoichiometry matrix
        reaction_index = 0
        for reaction,rate_constants in reactions_dict.items():
            update_matrix(reaction.reactants, reaction_index, -1)
            update_matrix(reaction.products, reaction_index, 1)
            if isinstance(rate_constants, tuple):
                reaction_index += 1
                update_matrix(reaction.reactants, reaction_index, 1)
                update_matrix(reaction.products, reaction_index, -1)
            reaction_index += 1
        
        return matrix

    def get_rate_vector(self, concentration_vector):
        """Compute the instantaneous rates given instantaneous concentrations.

        Args:
            concentration_vector (np.ndarray) : one-dimensional array of concentrations in order of self.species

        Returns:
            np.ndarray: (n_species,) rate of formation/destruction of each species
        """
        assert isinstance(concentration_vector, np.ndarray), f"expected np.array for concentration vector"
        n_species = len(self.species)
        assert concentration_vector.shape == (n_species,), f"expected shape of ({n_species},) in concentration vector but got shape {concentration_vector.shape}"
        
        # compute rate_reaction vector
        # the i-th element is the rate of reaction i, indexed by the order in self.reactions_dict
        # the rate of reaction i is the rate constant for that reaction multiplied by the product of all
        # concentrations on the left-hand side of the reaction (or the opposite for the reverse reactions)
        rate_reaction_vector = []
        for i,indices in enumerate(self.rate_indices):
            product = 1.0
            for j in indices:
                product *= concentration_vector[j]
            rate_reaction_vector.append(product)
        rate_reaction_vector = np.array(rate_reaction_vector) 
        rate_reaction_vector = self.rate_constant_vector * rate_reaction_vector

        # compute the rate for each species
        # the i-th element is the rate of production of species i,
        # indexed by the order in self.species 
        rate_species_vector = self.stoichiometry_matrix @ rate_reaction_vector

        # zero out entries for fixed concentrations
        rate_species_vector = self.fixed_concentrations_vector * rate_species_vector

        return rate_species_vector
