include("../../velocity/velocity.jl")

using DataFrames
using CSV
using Plots

hcolors = Dict(
    "red" => "#E63745",
    "blue" => "#276EBE",
    "lred" => "#F18F99",
    "lblue" => "#A8CAF7",
    "gray" => "#939393",
    "lgray" => "#D9D9D9",
)

function get_network_cat(params::Vector{Tuple{String,Float64,Float64}}, catalyst_death::Union{Float64,Bool}=false)
    """given a list of simple reactions in the form of a list of tuples (connection, base_rate, selectivity) where connection is a string in the form of "A_B" where A is a rectant and B is a product
    return a network of reactions

    Args:
        params (Vector{Tuple{String,Float64,Float64}}): list of parameters in the form of (connection, base_rate, selectivity)
        catalyst_death (Union{Float64,Bool}): if catalyst_death is a float then we add a reaction that kills the catalyst

    Returns:
        RxnNetwork: network of reactions
    """

    species = Species[]
    reactions = Reaction[]

    # loop through all params and add species and reactions
    # params are in the form of (connection, base_rate, selectivity)
    for (connection, base_rate, selectivity) in params
        # get reactant and product from connection
        reactant, product = split(connection, "_")

        # add species
        push!(species, Species(id=reactant))
        push!(species, Species(id=product))

        # add reaction
        push!(reactions, Reaction(reactants=Dict(Species(id=reactant) => 1), products=Dict(Species(id=product) => 1), catalysts=Dict(Species(id="Catalyst") => 1), rate_constant=get_rate_constants(base_rate, back_convert_sel(selectivity)), id=connection))
    end

    # add catalyst and dead catalyst species
    push!(species, Species(id="Catalyst"))
    push!(species, Species(id="Catalyst_Dead"))

    # if catalyst_death is a float then add a reaction that kills the catalyst
    if typeof(catalyst_death) == Float64
        push!(reactions, Reaction(reactants=Dict(Species(id="Catalyst") => 1), products=Dict(Species(id="Catalyst_Dead") => 1), rate_constant=(catalyst_death, 0), id="cat_death"))
    end

    # sort unique list of species
    species = sort(unique(species))

    # return network
    return RxnNetwork((reactions...,), (species...,))
end

function get_dataset(filename)
    """ get a formatted dataset from a csv file

    Args:
        filename (str): filename of csv file

    """
    # read data and group by run
    fit_data = CSV.read(filename, DataFrame)
    grouped = groupby(fit_data, :Run)
    sugars = ["Glc", "Man", "All", "Gal", "Alt", "Tal", "Gul", "Ido"]
    sugars = sort(sugars)

    datasets = []
    for g in grouped

        # sort time and get initial conditions
        sort!(g, :Time)
        initial_cond = g[g[!, :Time].==0.0, :]
        initial_cond = initial_cond[!, sugars] .* 0.2 .* 0.01 # convert to M
        initial_cond = [first(initial_cond[!, sugar]) for sugar in sugars]
        initial_cond_dict = Dict{Species,Float64}()
        for i in eachindex(sugars)
            initial_cond_dict[Species(id=sugars[i])] = initial_cond[i]
        end
        initial_cond_dict[Species(id="Catalyst")] = 0.004
        initial_cond_dict[Species(id="Catalyst_Dead")] = 0.0

        # get times and data excluding time 0
        times = Tuple(1.0 .* g[g[!, :Time].!=0.0, :Time])
        data = g[g[!, :Time].!=0.0, :]
        data = data[!, sugars] .* 0.2 .* 0.01 # convert to molarity

        # convert data to vector
        data_vector = []
        for row in eachrow(data)
            push!(data_vector, [row[sugar] for sugar in sugars])
        end
        push!(datasets, (initial_cond_dict, times, data_vector))
    end

    return datasets
end

function removecatconcs(dataset)
    """ remove catalyst concentrations from a simulation result """
    return [dataset[1:2]; dataset[5:end]]
end

function rmsd(error)
    """calculate the root mean square deviation of a vector of errors"""
    return sqrt(sum(error .^ 2) / length(error))
end

function rmsd(data, sim, damp=0)
    """calculate the root mean square deviation of two vectors
    optionally with a dampening factor
    
    Args:
        data (Vector): data to compare
        sim (Vector): simulation to compare
        damp (float): dampening factor for error (optional)
    
    Returns:
        float: root mean square deviation
    """
    # calculate error
    error = data .- sim

    # apply dampening if damp != 0
    if damp != 0
        error = [max(x - damp, 0) for x in error]
    end

    # return rmsd
    return sqrt(sum(error .^ 2) / length(data))
end

function loss_function(network, datasets)
    """ calculate the loss function for a given set of parameters and datasets

    Args:
        network (RxnNetwork): network to simulate
        datasets (Vector{(initial cond, timepoints, data)}): list of datasets
    """

    errors = []
    for dataset in datasets
        # simulate timecourse and get simulation data
        sim = simulate_timecourse(network, dataset[1], (0.0, maximum(dataset[2])), dataset[2]).u

        # calculate errors
        local_errors = []
        for (idx, timepoint) in enumerate(sim)
            timepoint = [timepoint[1:2]; timepoint[5:end]] # remove catalysts
            push!(local_errors, rmsd(dataset[3][idx], timepoint, 0))
        end
        push!(errors, rmsd(local_errors))
    end

    return rmsd(errors) / 0.2 * 100 # convert to percent
end

alpha_data = get_dataset("expt_data/alpha_alldata.csv")
beta_data = get_dataset("expt_data/beta_alldata.csv")

# parameters from fitting
alpha_params = [
    ("Glc_All", 8696.595466168013, 19.41682188288102),
    ("Glc_Gal", 2669.8514888247096, -1.0259999159948268),
    ("Glc_Man", 1700.4648769001128, -3.4503655195119873),
    ("All_Gul", 2272.2408829053693, -21.535708164494086),
    ("Gul_Gal", 9879.277501320068, 2.042525719539408),
    ("Gal_Tal", 1921.3481285089338, 21.44995804709182),
    ("Tal_Man", 3435.4755863963533, -21.71316524913186),
    ("Man_Alt", 8167.951419620186, -1.9331817271119984),
    ("Alt_All", 2251.332181358157, 21.505092729010048),
    ("Alt_Ido", 2243.2621555496744, -1.016210302510667),
    ("Tal_Ido", 3312.9198527517797, -21.611499469946136),
    ("Ido_Gul", 999.9938237781865, 10.614793727460544)
]
beta_params = [
    ("Glc_All", 15301.874158372651, 1.1046186389372123),
    ("Glc_Gal", 5406.052704709827, 1.1010294423827778),
    ("Glc_Man", 3081.0571102017593, 5.117632219503642),
    ("All_Gul", 4417.41878099147, -14.893229618571524),
    ("Gul_Gal", 7143.924691122882, 20.6151011423547),
    ("Gal_Tal", 8030.790054565239, 21.818684935483084),
    ("Tal_Man", 6800.291414624909, -21.580338807211955),
    ("Man_Alt", 5514.196193944934, -20.598720669500302),
    ("Alt_All", 2085.3079621791885, 14.222839220674421),
    ("Alt_Ido", 999.9940508958343, 7.109655829761684),
    ("Tal_Ido", 2090.6346900714825, -24.033691493134185),
    ("Ido_Gul", 1182.044361016479, 19.561375790185163),
]

# build networks
alpha_network = get_network_cat(alpha_params, 0.000014)
beta_network = get_network_cat(beta_params, 0.000014)

# print overall RMSD
println("Overall loss for alpha: ")
println(loss_function(alpha_network, alpha_data))
println("Overall loss for beta: ")
println(loss_function(beta_network, beta_data))

function get_pred_vs_expt_time_plots(network, dataset)
    """ get a list of plots comparing predicted vs experimental timecourses for a given dataset

    Args:
        network (RxnNetwork): network to simulate
        dataset ((initial cond, timepoints, data)): dataset to compare

    Returns:
        Vector{Plots.Plot}: list of plots
    """

    initial_cond, times, data_vector = dataset

    data_vector = [x ./ 0.2 for x in data_vector] # convert to mol%
    sim_data = simulate_timecourse(network, initial_cond, (0.0, maximum(times)), Tuple(range(1.0, maximum(times), length=100))).u
    sim_data = removecatconcs.(sim_data) ./ 0.2 # convert to mol % and remove catalysts

    sugars = sort(filter(x -> !contains(x, "Catalyst"), [x.id for x in network.species]))

    # sort graphs by maximum value
    sugar_order = sortperm([maximum([x[i] for x in data_vector]) for i in eachindex(sugars)], rev=true)

    plot_list = []
    for (num, idx) in enumerate(sugar_order)
        title = sugars[idx]

        # prepare ylim scaling
        max_val = maximum([x[idx] for x in data_vector])
        nearest_val = minimum([i for i in [0.1, 0.5, 1.0] if i >= max_val])
        ylim = (0, nearest_val)

        # plot with ribbon for +/- 2.5%
        p = plot(range(0, stop=maximum(times), length=100), [sim_data[i][idx] for i in eachindex(sim_data)]; color=hcolors["gray"], linecolor=hcolors["gray"], ribbon=0.025, title, titlefontcolor=hcolors["gray"], titlefontsize=8)
        scatter!(p, [times...], [x[idx] for x in data_vector]; color=hcolors["blue"], linecolor=hcolors["blue"], ylim=ylim)
        plot!(grid=false, frame=:box, framestyle=:bold, framecolor=hcolors["gray"])

        push!(plot_list, p)
    end

    return plot_list
end

# plot and save alpha data
for i in eachindex(alpha_data)
    a = plot(get_pred_vs_expt_time_plots(alpha_network, alpha_data[i])..., legend=false, layout=(2, 4), ms=4, lw=3, xticks=false, foreground_color_axis=hcolors["gray"], foreground_color_border=hcolors["gray"], foreground_color_text=hcolors["gray"], background_color=:transparent, size=(400, 180), dpi=800)
    savefig("alpha_$i.png")
end

# plot and save beta data
for i in eachindex(beta_data)
    a = plot(get_pred_vs_expt_time_plots(beta_network, beta_data[i])..., legend=false, layout=(2, 4), ms=4, lw=3, xticks=false, foreground_color_axis=hcolors["gray"], foreground_color_border=hcolors["gray"], foreground_color_text=hcolors["gray"], background_color=:transparent, size=(400, 180), dpi=800)
    savefig("beta_$i.png")
end