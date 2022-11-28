import numpy as np

class Species():
    """ This represents a single chemical species. """
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
    """ This represents a chemical reaction. """
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

