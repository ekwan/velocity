import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

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
        return f"Species ({self.abbreviation}={self.description})"

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
        #for s in reactants:
        #    assert s not in products, f"{s} is in both the reactants and products"

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
            assert isinstance(item, float), f"expected float for rate constant but got {item} ({type(item)}) for {reaction}"
            assert item >= 0.0, f"rate constant invalid for {reaction}"
            return True

        # check that all reactions and rate constants are valid
        for reaction,v in reactions_dict.items():
            assert isinstance(reaction, Reaction), f"expected Reaction but got {type(reaction)}"
            if reaction.reversible:
                assert isinstance(v, tuple), f"expected tuple for reversible rate constants but got {type(v)} for {reaction}"
                assert len(v) == 2, f"expected 2-tuple but got length {len(v)} for {reaction}"
                for rate_constant in v:
                    assert is_valid_rate_constant(rate_constant, reaction)
                    rate_constants.append(rate_constant)
            else:
                if isinstance(v, tuple) and len(v) == 1:
                    v = v[0]
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
            assert s in species, f"{s} is not in this reaction network"
        fixed_concentrations_vector = [0 if s in fixed_concentrations else 1 for s in species]

        # for each reaction, gather the index of each species and its stoichiometric coefficient
        # so that the rate reaction vector can be generated quickly
        # each entry is (index, stoichiometric coefficient)
        species_indices_and_coefficients = []
        for reaction in reactions_dict:
            indices_and_coefficients = [ (species.index(reactant), coefficient) for reactant, coefficient in reaction.reactants.items() ]
            species_indices_and_coefficients.append(indices_and_coefficients)
            if reaction.reversible:
                indices_and_coefficients = [ (species.index(product), coefficient) for product, coefficient in reaction.products.items() ]
                species_indices_and_coefficients.append(indices_and_coefficients)

        # store data
        self.species = species
        self.reactions_dict = reactions_dict
        self.rate_constant_vector = np.array(rate_constants)
        self.species_indices_and_coefficients = species_indices_and_coefficients
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
        # each entry is the number of molecules of the given species that are
        # created (positive entries) or destroyed (negative entries)
        def update_matrix(reaction, reaction_index, target, sign):
            assert isinstance(reaction, Reaction), f"unexpected type: {type(reaction)}"
            assert sign in [1,-1], "unknown sign"
            if target == "reactants":
                target = reaction.reactants
            elif target == "products":
                target = reaction.products
            else:
                raise ValueError(f"unknown target: {target}")

            for s,coefficient in target.items():
                species_index = all_species.index(s)
                matrix[species_index][reaction_index] += sign * coefficient

        # populate stoichiometry matrix
        reaction_index = 0
        for reaction,rate_constants in reactions_dict.items():
            # handle the forward reaction,
            # which takes away reactants and makes products
            update_matrix(reaction, reaction_index, "reactants", -1)
            update_matrix(reaction, reaction_index, "products", 1)

            # if reversible, handle the backward reaction,#
            # which takes away products and makes reactants
            if isinstance(rate_constants, tuple):
                reaction_index += 1
                update_matrix(reaction, reaction_index, "reactants", 1)
                update_matrix(reaction, reaction_index, "products", -1)
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
        for indices_and_coefficients in self.species_indices_and_coefficients:
            product = 1.0
            for index,coefficient in indices_and_coefficients:
                product *= concentration_vector[index] ** coefficient if coefficient > 1 else concentration_vector[index]
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

    def simulate_timecourse(self, initial_concentrations, t_span, t_eval=None, max_step=np.inf, method="RK45"):
        """ Runs a timecourse simulation of the specified system.

            Uses scipy.integrate.solve_ivp.

            Args:
                initial_concentrations (:obj:`dict`, list, or :obj:`np.ndarray`):
                                           { Species : concentration (float) }, any Species not included
                                           are assumed to begin at a concentration of zero, or array of concentrations
                                           indexed by self.Species
                t_span (tuple): (t_start, t_finish)
                t_eval (None or np.ndarray): times at which to store the computed solution, must be sorted
                                             and lie within t_span, defaults to None
                max_step (float): maximum time step, defaults to np.inf
                method (str): see documentation for scipy.integrate.solve_ivp, defaults to "RK45"

            Returns:
                concentrations_df: rows are times, columns are species
        """

        # check inputs and convert them to a concentration vector
        if isinstance(initial_concentrations, dict):
            for k,v in initial_concentrations.items():
                assert isinstance(k, Species), f"expected Species got {type(k)}"
                assert k in self.species, f"{k} not in this Network"
                assert v >= 0.0, f"concentration of {k} must be non-negative"
            initial_concentrations = [initial_concentrations[s] if s in initial_concentrations else 0.0 for s in self.species]
            assert np.sum(initial_concentrations) > 0.0, "must have at least one non-zero initial concentration"
        if isinstance(initial_concentrations, (list,np.ndarray)):
            if isinstance(initial_concentrations, list):
                initial_concentrations = np.array(initial_concentrations, dtype=np.float64)
            n_species = len(self.species)
            assert initial_concentrations.shape == (n_species,), f"expected {len(self.species)}-length vector for initial concentrations, but got shape {initial_concentrations.shape}"
            for i in initial_concentrations:
                assert i >= 0.0, f"all concentrations in initial_concentrations must be non-negative:\n{initial_concentrations}"
        else:
            raise ValueError(f"Expected dictionary, list, or np.ndarray for initial concentrations but got {type(initial_concentrations)}.")

        assert isinstance(t_span, tuple), f"expected times to be given as a tuple, got {type(t_span)} instead"
        assert len(t_span) == 2, f"tuple should be length 2"
        if t_eval is not None:
            if isinstance(t_eval, (int,float)):
                t_eval = [t_eval]
            if isinstance(t_eval, list):
                t_eval = np.array(t_eval)
            assert isinstance(t_eval, np.ndarray), f"expected evaluation times to be np.ndarray but got {type(t_eval)} instead"

        assert isinstance(method, str), f"expected str for method but got {type(method)}"

        assert isinstance(max_step, float) and max_step > 0.0, f"invalid max timestep: {max_step}"

        # run simulation
        f = lambda t,y : self.get_rate_vector(y)
        solution = solve_ivp(f, t_span, initial_concentrations, t_eval=t_eval, max_step=max_step, method=method)

        # convert to dataframe
        concentrations_dict = { s.abbreviation : solution.y[i,:] for i,s in enumerate(self.species) }
        concentrations_df = pd.DataFrame(concentrations_dict)
        concentrations_df.index = solution.t
        concentrations_df.index.name = "time"
        return concentrations_df

class Addition():
    """Represents an addition of various Species to an Experiment.

    This includes the addition at t=0.

    Attributes:
        experiment (Experiment): Which experiment this Addition is attached to.
        concentrations_dict (dict): { Species : concentration (float) }, the concentrations in the addition.
                                    Species that are not included will be assumed to have no additional concentration.
        volume (float): The volume of the aliquot to add, in the same arbitrary units as experiment.initial_volume.
    """
    def __init__(self, experiment, concentrations_dict, volume):
        assert isinstance(experiment, Experiment)

        assert isinstance(concentrations_dict, dict)
        for species,concentration in concentrations_dict.items():
            assert isinstance(species, Species)
            assert species in experiment.network.species
            assert isinstance(concentration, float)
            assert concentration >= 0.0
        self.concentrations_dict = concentrations_dict

        if volume != None:
            if isinstance(volume, int):
                volume = float(volume)
            assert isinstance(volume, float)
            assert volume > 0.0
        self.volume = volume

    def __repr__(self):
        return f"Addition (volume={self.volume}, {str(self.concentrations_dict)})"

class Experiment():
    """Represents a kinetics experiment.

    You can simulate the timecourse of the experiment with the addition of more reagents
    and calculate a loss vs. observed concentrations.

    Attributes:
        network (Network): The reaction network.
        initial_concentrations_dict (dict): { Species : concentration (float) } at the beginning of the experiment.
        max_time (float): When the end of the experiment is in seconds.
        initial_volume (float, optional): The initial volume of the vessel (in arbitrary units).
        eval_times (list or np.array, optional): Times when we want explicit concentrations to be calculated (default=[]).

        additions (list): [ (time (in seconds, float), Addition), ... ]
        observations (dict): { time (in seconds, float) : [ concentrations (float) in network.species order ] }
    """
    def __init__(self, network, initial_concentrations_dict, max_time, initial_volume=None, eval_times=[]):
        assert isinstance(network, Network)
        self.network = network

        if initial_volume is not None:
            if isinstance(initial_volume, int):
                initial_volume = float(initial_volume)
            assert isinstance(initial_volume, float)
            assert initial_volume > 0.0
        self.initial_volume = initial_volume

        # setup the initial addition of reagents
        initial_addition = Addition(self, initial_concentrations_dict, initial_volume)
        self.initial_concentrations_dict = initial_concentrations_dict
        self.additions = [(0.0,initial_addition)]

        assert isinstance(max_time, (float, int))
        assert max_time > 0.0
        self.max_time = float(max_time)

        if isinstance(eval_times, np.ndarray):
            assert len(eval_times.shape) == 1
            eval_times = list(eval_times)
        assert isinstance(eval_times, list)
        for i,t in enumerate(eval_times):
            assert 0.0 <= t <= max_time
            if i > 0:
                assert t > eval_times[i-1]
        self.eval_times = eval_times

    def schedule_addition(self, when, what, volume):
        """Schedule the addition of reagents.

        Additions must be scheduled in ascending chronological order.

        Args:
            when (float): When to add the reagents (in seconds).
            what: What to add: { Species : concentration (float) }.
            volume: The volume of the aliquot.
        """
        assert self.initial_volume is not None, "cannot schedule additions if the initial volume is not specified"
        assert isinstance(when, (float, int))
        assert when > 0
        if len(self.additions) > 0:
            last_addition = self.additions[-1]
            last_time = last_addition[0]
            assert last_time < when < self.max_time

        assert isinstance(what, dict)

        if isinstance(volume, int):
            volume = float(volume)
        assert isinstance(volume, float)
        assert volume > 0.0
        self.volume = volume
        addition = Addition(self, what, volume)

        self.additions.append((when, addition))

    def simulate(self):
        """Numerically integrate the rate equations, incorporating any scheduled additions.

        Sets self.df (pd.DataFrame).  Columns are species abbreviations, index is time.
        """

        # contain the concentrations vs. time data as one df per segment
        dfs = []

        # split the simulation into different segments
        #
        # where t_span = (start_time, end_time)
        # and t_eval = [start_time, intermediate_times ..., end_time] 
        volume = self.initial_volume

        for i,addition_tuple in enumerate(self.additions):
            # pull out the times for this segment
            segment_start_time,addition = addition_tuple
            segment_end_time = self.additions[i+1][0] if i < len(self.additions) - 1 else self.max_time
            t_span = (segment_start_time, segment_end_time)

            # ensure that we get back the concentrations at the start and end of the segment
            t_eval = [ t for t in self.eval_times if segment_start_time <= t < segment_end_time ]
            if self.eval_times[-1] == segment_end_time:
                t_eval.append(self.eval_times[-1])

            if volume:
                # volumes have been provided
                if i == 0:
                    # this is the initial charge
                    concentrations = addition.concentrations_dict
                else:
                    # these are additions, so go from concentration and volume to moles
                    current_moles = { species_abbreviation : concentration*volume 
                                      for species_abbreviation,concentration in concentrations.items() }
                    additional_moles = { species_abbreviation : concentration*addition.volume 
                                         for species_abbreviation,concentration in addition.concentrations_dict.items() }

                    # add up the moles
                    for s,m in additional_moles.items():
                        if s in current_moles:
                            current_moles[s] += m
                        else:
                            current_moles[s] = m

                    # divide by total volume to get concentrations again
                    volume += addition.volume
                    for s,m in current_moles.items():
                        current_moles[s] = m/volume
                    concentrations = current_moles
            else:
                # this is a single addition with no provided volume, so don't dilute
                concentrations = addition.concentrations_dict

            # run the simulation
            print("---")
            print(f"{concentrations=}")
            print(f"{t_span=}")
            print(f"{t_eval=}")
            print("---")
            df = self.network.simulate_timecourse(concentrations, t_span, t_eval)
            
            # store the result
            dfs.append(df)
            last_row = df.iloc[-1]
            concentrations = { self.network.species[i] : concentration for i,concentration in enumerate(last_row) }
            #print(concentrations)

        # combine results
        self.df = pd.concat(dfs)





    def add_observation(self, time, concentrations_dict):
        pass