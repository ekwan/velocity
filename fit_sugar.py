import pandas as pd
import numpy as np
from velocity.chemistry import Species, Reaction, Network
from math import sqrt
import lmfit, tqdm

# settings
observations_filename = "observations.xlsx"  # spreadsheet filename
mode = "simple"                              # whether to include the intermediate in the
                                             # kinetic model: "simple" or "complex"

# for all models
min_log10_base_rate_constant, default_log10_base_rate_constant, max_log10_base_rate_constant = -5, -3, -1
min_log10_catalyst_deactivation_rate_constant, default_log10_catalyst_deactivation_rate_constant, max_log10_catalyst_deactivation_rate_constant = -7, -5, -3

# for mode == "simple"
min_log10_overall_selectivity, default_log10_overall_selectivity, max_log10_overall_selectivity = -1, 0, 1

# for mode == "complex"
k_to_j_ratio = 1000.0                        # how downhill every turn of the ratchet is
min_log10_ablation_selectivity, default_log10_ablation_selectivity, max_log10_ablation_selectivity = -3, 0, 3
min_log10_regeneration_selectivity, default_log10_regeneration_selectivity, max_log10_regeneration_selectivity = -3, 0, 3

########

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
sugars_dict = { s.abbreviation : s for s in sugars }
sugar_abbreviations = [ s.abbreviation for s in sugars ]

# setup radicals
intermediates = [ Species(f"I{i+1}") for i in range(12) ]

# setup catalyst
catalyst_active = Species("cat", "catalyst (active)")
catalyst_dead = Species("dead", "catalyst (dead)")

# these are the network connections, with the numbers representing the intermediates
CONNECTIONS = ["Glc_1_All",  "Glc_2_Gal",  "Glc_3_Man",
               "All_4_Gul",  "Gul_5_Gal",  "Gal_6_Tal",
               "Tal_7_Man",  "Man_8_Alt",  "Alt_9_All",
               "Alt_10_Ido", "Tal_11_Ido", "Ido_12_Gul"]

# create parameters to be fit
assert mode in ["simple", "complex"]
parameters = lmfit.Parameters()
for connection in CONNECTIONS:
    parameter = lmfit.Parameter(
        name = f"{connection}_log10_base_rate_constant",
        min = min_log10_base_rate_constant,
        value = default_log10_base_rate_constant,
        max = max_log10_base_rate_constant, 
    )
    parameters.add(parameter)

    if mode == "simple":
        parameter = lmfit.Parameter(
            name = f"{connection}_log10_overall_selectivity",
            min = min_log10_overall_selectivity,
            value = default_log10_overall_selectivity,
            max = max_log10_overall_selectivity, 
        )
        parameters.add(parameter)

    elif mode == "complex":
        parameter = lmfit.Parameter(
            name = f"{connection}_log10_ablation_selectivity",
            min = min_log10_ablation_selectivity,
            value = default_log10_ablation_selectivity,
            max = max_log10_ablation_selectivity, 
        )
        parameters.add(parameter)

        parameter = lmfit.Parameter(
            name = f"{connection}_log10_regeneration_selectivity",
            min = min_log10_regeneration_selectivity,
            value = default_log10_regeneration_selectivity,
            max = max_log10_regeneration_selectivity, 
        )
        parameters.add(parameter)

parameter = lmfit.Parameter(
    name = f"log10_catalyst_deactivation_rate_constant",
    min = min_log10_catalyst_deactivation_rate_constant,
    value = default_log10_catalyst_deactivation_rate_constant,
    max = max_log10_catalyst_deactivation_rate_constant, 
)
parameters.add(parameter)
parameters.pretty_print()
print(f"There are {len(parameters)} parameters.")

# represents a single experimental isomerization run
class ExperimentalRun():
    def __init__(self, starting_sugar):
        assert starting_sugar in sugar_abbreviations, f"sugar {starting_sugar} not found"
        self.starting_sugar = sugars_dict[starting_sugar]
        self.observation_times = []          # list of floats
        self.observations = []               # list of np.array (float),
                                             # where the outer index parallels observation_times
                                             # and the inner index parallels sugars

    def add_observation(self, observation_time, observed_mole_fractions):
        assert isinstance(observation_time, (int,float)) and observation_time > 0.0
        self.observation_times.append(float(observation_time))
        assert isinstance(observed_mole_fractions, list)
        assert len(observed_mole_fractions) == len(sugars)
        for i in observed_mole_fractions:
            assert isinstance(i, (int,float)) and i >= 0.0
        observed_mole_fractions = np.array(observed_mole_fractions, dtype=float)
        observed_mole_fractions = observed_mole_fractions / np.sum(observed_mole_fractions)
        self.observations.append(observed_mole_fractions)

# read experimental data
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
        current_run = ExperimentalRun(starting_sugar)
        current_run_number = run_number
    elif run_number != current_run_number:
        assert run_number == current_run_number + 1
        experimental_runs.append(current_run)
        current_run = ExperimentalRun(starting_sugar)
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
print(f"Read {len(experimental_runs)} experimental runs.")
print(f"Will evaluate these times within {t_span} s:")
print(all_times)

###########

# adds the specified connection to the network
# connection_string: sugar1_intermediate_index_sugar2
# sugar1, sugar2 are abbreviations as str
# intermediate_index is 1-indexed int
# ablation_selectivity and regeneration_selectivity are used for mode="complex"
# overall_selectivity is used for mode="simple"
def add(reactions_dict, connection_string, base_rate_constant,
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

def create_network(x):
    reactions_dict = {}

    if mode == "simple":
        for connection in CONNECTIONS:
            base_rate_constant = np.power(10, x[f"{connection}_log10_base_rate_constant"])
            overall_selectivity = np.power(10, x[f"{connection}_log10_overall_selectivity"])
            add(reactions_dict, connection, base_rate_constant,
                overall_selectivity=overall_selectivity)
    elif mode == "complex":
        for connection in CONNECTIONS:
            base_rate_constant = np.power(10, x[f"{connection}_log10_base_rate_constant"])
            ablation_selectivity = np.power(10, x[f"{connection}_log10_ablation_selectivity"])
            regeneration_selectivity = np.power(10, x[f"{connection}_log10_regeneration_selectivity"])
            add(reactions_dict, connection, base_rate_constant,
                ablation_selectivity=ablation_selectivity, regeneration_selectivity=regeneration_selectivity)

    catalyst_deactivation_reaction = Reaction(catalyst_active, catalyst_dead, reversible=False)
    reactions_dict[catalyst_deactivation_reaction] = np.power(10,x["log10_catalyst_deactivation_rate_constant"])
    
    network = Network(reactions_dict, fixed_concentrations=None)
    return network

# root mean square average
def rms(x):
    assert isinstance(x, (list,np.ndarray))
    assert len(x) > 0
    return np.sqrt(np.mean(np.square((x))))

# simulate each experimental run for one set of parameters
def trial(x):
    network = create_network(x)

    # run simulations
    losses = []
    #x.pretty_print()
    for i,run in enumerate(experimental_runs, start=1):
        initial_concentrations_dict = {
            run.starting_sugar : 1.0,
            catalyst_active : 1.0
        }
        concentrations_df = network.simulate_timecourse(initial_concentrations_dict, t_span, t_eval)
        loss = loss_function(run, concentrations_df)
        print(f"{loss:8.4f} ", end="", flush=True)
        losses.append(loss)

    # aggregate losses
    loss = rms(losses)

    #
    print(f"  ::: {loss:8.4f}", flush=True)
    return loss

# calculate the loss for a single simulation
# treats all runs with same weight regardless of number of observations
def loss_function(experimental_run, concentrations_df):
    losses = []
    for t,observed in zip(experimental_run.observation_times, experimental_run.observations):
        df = concentrations_df.query(f"index == {t}")
        assert len(df) == 1
        simulated = concentrations_df.tail(1)[sugar_abbreviations].iloc[0].to_numpy()
        # print("Simulated:")
        # print(simulated)
        # print("Obeserved:")
        # print(observed)
        # print("diff:")
        # print(simulated-observed)
        # print("rms:")
        loss = rms(simulated-observed)
        # print(loss)
        losses.append(loss)
    # print("LOSSES")
    # print(losses)
    # print(rms(losses))
    return rms(losses)

# run the optimization
def iter_cb(parameters, iteration, residual):
    print(f"iteration {iteration:5d}   loss = {residual:12.4f}", flush=True)
output = lmfit.minimize(trial, parameters, method="differential_evolution", iter_cb=iter_cb)
