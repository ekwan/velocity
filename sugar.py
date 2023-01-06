import pandas as pd
import numpy as np
from velocity.chemistry import Species, Reaction, Network
from math import sqrt
import lmfit

# these are the network connections, with the numbers representing the intermediates
CONNECTIONS = ["Glc_1_All",  "Glc_2_Gal",  "Glc_3_Man",
               "All_4_Gul",  "Gul_5_Gal",  "Gal_6_Tal",
               "Tal_7_Man",  "Man_8_Alt",  "Alt_9_All",
               "Alt_10_Ido", "Tal_11_Ido", "Ido_12_Gul"]

def setup_species():
    # setup sugars
    glucose = Species("Glc", "glucose")
    mannose = Species("Man", "mannose")
    altrose = Species("Alt", "altrose")
    allose = Species("All", "allose")
    gulose = Species("Gul", "gulose")
    galactose = Species("Gal", "galactose")
    talose = Species("Tal", "talose")
    idose = Species("Ido", "idose")

    # must match order in spreadsheet!
    sugars = [ glucose, mannose, allose, galactose, altrose, talose, gulose, idose ]

    # setup radicals
    intermediates = [ Species(f"I{i+1}") for i in range(12) ]

    # setup catalyst
    catalyst_active = Species("cat", "catalyst (active)")
    catalyst_dead = Species("dead", "catalyst (dead)")

    return sugars, intermediates, catalyst_active, catalyst_dead

# adds the specified connection to the network
# connection_string: sugar1_intermediate_index_sugar2
# sugar1, sugar2 are abbreviations as str
# intermediate_index is 1-indexed int
# ablation_selectivity and regeneration_selectivity are used for mode="complex"
# overall_selectivity is used for mode="simple"
def add(reactions_dict, mode,
        sugars_dict, intermediates, catalyst_active,
        connection_string, base_rate_constant,
        overall_selectivity=None,
        ablation_selectivity=None, regeneration_selectivity=None):
    sugar1, intermediate_index, sugar2 = connection_string.split("_")
    sugar1 = sugars_dict[sugar1]
    intermediate = intermediates[int(intermediate_index)-1]
    sugar2 = sugars_dict[sugar2]

    if mode == "complex":
        j = base_rate_constant/sqrt(k_to_j_ratio)
        k = base_rate_constant*sqrt(k_to_j_ratio)

        j1 = sqrt(ablation_selectivity)*j
        j2 = j/sqrt(ablation_selectivity)
        k1 = sqrt(regeneration_selectivity)*k
        k2 = k/sqrt(regeneration_selectivity)
        
        reaction1 = Reaction({sugar1:1, catalyst_active:1}, {intermediate:1, catalyst_active:1}, reversible=True)
        reaction2 = Reaction({intermediate:1, catalyst_active:1}, {sugar2:1, catalyst_active:1}, reversible=True)
        reactions_dict.update({
            reaction1 : (j1,k2),
            reaction2 : (k1,j2),
        })
    elif mode == "simple":
        jk1 = base_rate_constant*sqrt(overall_selectivity)
        jk2 = base_rate_constant/sqrt(overall_selectivity)
        reaction = Reaction({sugar1:1, catalyst_active:1}, {sugar2:1, catalyst_active:1}, reversible=True)
        reactions_dict[reaction] = (jk1,jk2)

def convert(selectivity):
    selectivity= float(selectivity)
    if selectivity < 0.0:
        selectivity = -1/selectivity
    return selectivity

def create_network(parameters, sugars, intermediates, catalyst_active, catalyst_dead):
    reactions_dict = {}
    x = parameters
    sugars_dict = { s.abbreviation : s for s in sugars }
    mode = parameters["model_type"]
    conversion=1e-6

    if mode == "simple":
        for connection in CONNECTIONS:
            base_rate_constant = float(x[f"{connection}_base"])*conversion 
            overall_selectivity = convert(x[f"{connection}_overall"])
            add(reactions_dict, mode,
                sugars_dict, intermediates, catalyst_active,
                connection, base_rate_constant,
                overall_selectivity=overall_selectivity)
    elif mode == "complex":
        for connection in CONNECTIONS:
            base_rate_constant = float(x[f"{connection}_base"])*conversion 
            ablation_selectivity = convert(x[f"{connection}_ablation"]) 
            regeneration_selectivity = convert(x[f"{connection}_regeneration"]) 
            add(reactions_dict, mode,
                sugars_dict, intermediates, catalyst_active,
                connection, base_rate_constant,
                ablation_selectivity=ablation_selectivity, regeneration_selectivity=regeneration_selectivity)

    catalyst_deactivation_reaction = Reaction(catalyst_active, catalyst_dead, reversible=False)
    reactions_dict[catalyst_deactivation_reaction] = x["catalyst_deactivation"]*conversion #np.power(10,x["log10_catalyst_deactivation_rate_constant"])
    
    network = Network(reactions_dict, fixed_concentrations=None)
    return network

# represents a single experimental isomerization run
class ExperimentalRun():
    def __init__(self, sugars, starting_sugar):
        sugars_dict = { s.abbreviation : s for s in sugars }
        assert starting_sugar in sugars_dict, f"sugar {starting_sugar} not found"
        self.sugars = sugars
        self.starting_sugar = sugars_dict[starting_sugar]
        self.observation_times = []          # list of floats
        self.observations = []               # list of np.array (float),
                                             # where the outer index parallels observation_times
                                             # and the inner index parallels sugars

    def add_observation(self, observation_time, observed_mole_fractions):
        assert isinstance(observation_time, (int,float)) and observation_time > 0.0
        self.observation_times.append(float(observation_time))
        assert isinstance(observed_mole_fractions, list)
        assert len(observed_mole_fractions) == len(self.sugars)
        for i in observed_mole_fractions:
            assert isinstance(i, (int,float)) and i >= 0.0
        observed_mole_fractions = np.array(observed_mole_fractions, dtype=float)
        observed_mole_fractions = observed_mole_fractions / np.sum(observed_mole_fractions)
        self.observations.append(observed_mole_fractions)

def read_experimental_runs(observations_filename, sugars, verbose=False):
    sugar_abbreviations = {s.abbreviation : s for s in sugars }
    observations_df = pd.read_excel(observations_filename)
    expected_columns = ["Run", "Starting Sugar", "Time"]
    expected_columns.extend(sugar_abbreviations)
    assert np.all(observations_df.columns == expected_columns), "check spreadsheet columns"
    current_run = None
    current_run_number = None
    experimental_runs = []
    for _,row in observations_df.iterrows():
        run_number, starting_sugar, time, *observations = row
        if current_run is None:
            assert run_number == 1
            current_run = ExperimentalRun(sugars, starting_sugar)
            current_run_number = run_number
        elif run_number != current_run_number:
            assert run_number == current_run_number + 1
            experimental_runs.append(current_run)
            current_run = ExperimentalRun(sugars, starting_sugar)
            current_run_number = run_number
        else:
            assert starting_sugar == current_run.starting_sugar.abbreviation
        current_run.add_observation(time, observations)

    if current_run:
        experimental_runs.append(current_run)

    # generate times to run simulation over
    all_times = set()
    for r in experimental_runs:
        all_times.update(r.observation_times)
        # print(r.starting_sugar.abbreviation)
        # print(r.observation_times)
        # print(r.observations)
        # print()
    all_times = list(sorted(all_times))
    t_span = (0.0, all_times[-1])
    t_eval = np.array(all_times)

    if verbose:
        print(f"Read {len(experimental_runs)} experimental runs.")
        print(f"Will evaluate these times within {t_span} s:")
        print(all_times)

    return experimental_runs, t_span, t_eval



def simulate(network, catalyst_active, experimental_runs, t_span, t_eval):
    # run simulations
    results = []
    for i,run in enumerate(experimental_runs, start=1):
        initial_concentrations_dict = {
            run.starting_sugar : 1.0,
            catalyst_active : 1.0
        }
        concentrations_df = network.simulate_timecourse(initial_concentrations_dict, t_span, t_eval)
        #print(concentrations_df)
        #print()
        results.append(concentrations_df)
    return results



