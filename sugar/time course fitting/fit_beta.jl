include("../../velocity/velocity.jl")

using DataFrames
using CSV
using LinearAlgebra
using Distributions
using Metaheuristics
import Optim

sugars = ["Glc", "Man", "All", "Gal", "Alt", "Tal", "Gul", "Ido"]
sugars = sort(sugars)
connections = [
    "Glc_All",
    "Glc_Gal",
    "Glc_Man",
    "All_Gul",
    "Gul_Gal",
    "Gal_Tal",
    "Tal_Man",
    "Man_Alt",
    "Alt_All",
    "Alt_Ido",
    "Tal_Ido",
    "Ido_Gul",
]

# settings for DE fitting
rate_bounds = [1000, 30000]
sel_bounds = [-20, 20]


# import and process training data
fit_data = CSV.read("expt_data/beta_trainset.csv", DataFrame)

grouped = groupby(fit_data, :Run)
datasets = []
for g in grouped
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

    times = Tuple(1.0 .* g[g[!, :Time].!=0.0, :Time])

    data = g[g[!, :Time].!=0.0, :]
    data = data[!, sugars] .* 0.2 .* 0.01
    data_vector = []

    for row in eachrow(data)
        push!(data_vector, [row[sugar] for sugar in sugars])
    end

    push!(datasets, (initial_cond_dict, times, data_vector))
end


function rmsd(error)
    return sqrt(sum(error .^ 2) / length(error))
end

function rmsd(data, sim, damp=0)
    error = data .- sim
    if damp != 0
        error = [max(x - damp, 0) for x in error]
    end

    return sqrt(sum(error .^ 2) / length(data))
end

function fixsel(sel)
    # we use fixsel to have a continuous domain for the selectivity fitting
    if sel < 0
        return sel - 1
    else
        return sel + 1
    end
end

function get_network_cat(params::Vector{Tuple{String,Float64,Float64}}, catalyst_death::Union{Float64,Bool}=false)
    # given a list of simple reactions in the form of a list of tuples (connection, base_rate, selectivity) where connection is a string in the form of "A_B" where A is a rectant and B is a products
    # return a network of reactions

    species = Species[]
    reactions = Reaction[]
    for (connection, base_rate, selectivity) in params
        reactant, product = split(connection, "_")
        push!(species, Species(id=reactant))
        push!(species, Species(id=product))
        push!(reactions, Reaction(reactants=Dict(Species(id=reactant) => 1), products=Dict(Species(id=product) => 1), catalysts=Dict(Species(id="Catalyst") => 1), rate_constant=get_rate_constants(base_rate, back_convert_sel(selectivity)), id=connection))
    end

    push!(species, Species(id="Catalyst"))
    push!(species, Species(id="Catalyst_Dead"))

    # if catalyst_death is a float then we add a reaction that kills the catalyst
    if typeof(catalyst_death) == Float64
        push!(reactions, Reaction(reactants=Dict(Species(id="Catalyst") => 1), products=Dict(Species(id="Catalyst_Dead") => 1), rate_constant=(catalyst_death, 0), id="cat_death"))
    end

    species = unique(species)
    return RxnNetwork((reactions...,), (species...,))
end


function loss_function(paramset)

    # we use fixsel to have a continuous domain for the selectivity fitting
    paramset_formatted = [(connections[i], paramset[i], fixsel(paramset[i+length(connections)])) for i in eachindex(connections)]

    # 0.000014 is the rate of catalyst degradation
    paramset_formatted_float = [(x[1], float(x[2]), float(x[3])) for x in paramset_formatted]
    network = get_network_cat(paramset_formatted_float, 0.000014)

    errors = []

    for dataset in datasets
        sim = simulate_timecourse(network, dataset[1], (0.0, maximum(dataset[2])), dataset[2]).u
        local_errors = []
        for (idx, timepoint) in enumerate(sim)
            timepoint = [timepoint[1:2]; timepoint[5:end]] # remove catalysts
            push!(local_errors, rmsd(dataset[3][idx], timepoint, 0))
        end
        push!(errors, rmsd(local_errors))
    end

    return rmsd(errors) / 0.2 * 100 # convert to percent
end

#setup bounds for DE
parameter_bounds = [[rate_bounds for i in 1:length(connections)]; [sel_bounds for i in 1:length(connections)]]
lb = Float64[x[1] for x in parameter_bounds]
ub = Float64[x[2] for x in parameter_bounds]
bounds = boxconstraints(lb=lb, ub=ub)

#setup parallel fitting function
function f_parallel(X)
    fitness = zeros(size(X, 1))
    Threads.@threads for i in 1:size(X, 1)
        fitness[i] = loss_function(X[i, :])
    end
    return fitness
end

#setup DE
options = Options(f_calls_limit=10000000, f_tol=1e-3, seed=1, parallel_evaluation=true)
algorithm = DE(options=options)

#run fitting + print results
result = optimize(f_parallel, bounds, algorithm)
println(result)
minimum_params = minimizer(result)
println(minimum_params)

#print formatted rate constants
println("Formated rate constants")
paramset_formatted = [(connections[i], minimum_params[i], fixsel(minimum_params[i+length(connections)])) for i in eachindex(connections)]
for x in paramset_formatted
    println(x)
end

#run LBFGS on the best result from DE
results2 = Optim.optimize(loss_function, minimum_params, Optim.LBFGS(), Optim.Options(g_tol=1e-3, iterations=1000))

# print results
println(Optim.summary(results2))
println(Optim.minimizer(results2))

#print formatted rate constants
paramset_formatted = [(connections[i], Optim.minimizer(results2)[i], fixsel(Optim.minimizer(results2)[i+length(connections)])) for i in eachindex(connections)]
println("Formated rate constants")
for x in paramset_formatted
    println(x)
end
println(Optim.minimum(results2))