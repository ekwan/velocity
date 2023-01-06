import pandas as pd
import numpy as np
from velocity.chemistry import Species, Reaction, Network
from math import sqrt
import lmfit, tqdm

# settings
observations_filename = "observations.xlsx"  # spreadsheet filename

# parameter bounds
min_log10_base_rate_constant = -6
max_log10_base_rate_constant = -3
min_log10_overall_selectivity = -1
max_log10_overall_selectivity = 1
min_log10_catalyst_deactivation_rate_constant = -6
default_log10_catalyst_deactivation_rate_constant = np.log10(50*1e-6)
max_log10_catalyst_deactivation_rate_constant = -4

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
parameters = lmfit.Parameters()
def add_parameters(i, base_rate_constant, selectivity, vary=True):
    connection = CONNECTIONS[i-1]

    parameter = lmfit.Parameter(
        name = f"{connection}_log10_base_rate_constant",
        min = min_log10_base_rate_constant,
        value = np.log10(base_rate_constant*1e-6),
        max = max_log10_base_rate_constant,
        vary = vary
    )
    parameters.add(parameter)

    if selectivity < 0:
        selectivity = -1/selectivity
    parameter = lmfit.Parameter(
        name = f"{connection}_log10_overall_selectivity",
        min = min_log10_overall_selectivity,
        value = np.log10(selectivity),
        max = max_log10_overall_selectivity,
        vary = vary 
    )
    parameters.add(parameter)

add_parameters(1,  300, 10, False)
add_parameters(2,  23,   1)
add_parameters(3,  58,   1)
add_parameters(4,  100,   1)
add_parameters(5,   30,   1)
add_parameters(6,  20,  10, False)
add_parameters(7,  50, -5, False)
add_parameters(8,  5,   1)
add_parameters(9,  50,  1)
add_parameters(10, 7,   1)
add_parameters(11, 10,   1)
add_parameters(12, 90,   1)

parameter = lmfit.Parameter(
    name = f"log10_catalyst_deactivation_rate_constant",
    min = min_log10_catalyst_deactivation_rate_constant,
    value = default_log10_catalyst_deactivation_rate_constant,
    max = max_log10_catalyst_deactivation_rate_constant,
    vary = False
)
parameters.add(parameter)
parameters.pretty_print()
print(f"There are {len(parameters)} parameters.\n")

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
print()

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

    jk1 = base_rate_constant*sqrt(overall_selectivity)
    jk2 = base_rate_constant/sqrt(overall_selectivity)
    reaction = Reaction({sugar1:1, catalyst_active:1}, {sugar2:1, catalyst_active:1}, reversible=True)
    reactions_dict[reaction] = (jk1,jk2)

def create_network(x):
    reactions_dict = {}

    for connection in CONNECTIONS:
        base_rate_constant = np.power(10, x[f"{connection}_log10_base_rate_constant"])
        overall_selectivity = np.power(10, x[f"{connection}_log10_overall_selectivity"])
        add(reactions_dict, connection, base_rate_constant,
            overall_selectivity=overall_selectivity)

    catalyst_deactivation_reaction = Reaction(catalyst_active, catalyst_dead, reversible=False)
    reactions_dict[catalyst_deactivation_reaction] = np.power(10.0,x["log10_catalyst_deactivation_rate_constant"])
    
    network = Network(reactions_dict, fixed_concentrations=None)
    return network

# root mean square average
def rms(x):
    assert isinstance(x, (list,np.ndarray))
    assert len(x) > 0
    return np.sqrt(np.mean(np.square((x))))

# simulate each experimental run for one set of parameters
iteration=0
def trial(x):
    global iteration
    iteration += 1
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
    print(f"  ::: {loss:8.4f}     (iteration={iteration})", end="\r", flush=True)
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
results = lmfit.minimize(trial, parameters, method="nelder", max_nfev=10)
print()
print(lmfit.fit_report(results))
print()
print_count = 0
for name, param in results.params.items():
    print_count += 1
    value = np.power(10.0, param.value)
    if "rate" in name:
        value = value*1e6
    if print_count % 2:
        print(f"{name:42s} {value:10.0f}   ", end="")
    else:
        if value < 1:
            value = -1/value
        print(f"{value:10.0f}")
print()