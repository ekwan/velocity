include("../../velocity/velocity.jl")

using DataFrames
using CSV
using LinearAlgebra
using Distributions
using Plots

sugars = ["Glc", "Man", "All", "Gal", "Alt", "Tal", "Gul", "Ido"]
sugars = sort(sugars)

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
connections = [x[1] for x in alpha_params]

sel_set = [-20:-2; 1:20]
rate_set = 1000:500:20000


# import and process training data
fit_data = CSV.read("../expt_data/alpha_trainset.csv", DataFrame)
grouped = groupby(fit_data, :Run)
sugars = ["Glc", "Man", "All", "Gal", "Alt", "Tal", "Gul", "Ido"]
sugars = sort(sugars)
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
    # 0.000014 is the rate of catalyst degradation
    network = get_network_cat(paramset, 0.000014)

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

    return rmsd(errors) / 0.2 * 100, maximum(errors) / 0.2 * 100 # convert to percent
end

function limitto(num, limit)
    if num > limit
        return NaN
    else
        return num
    end
end

# create a dictionary to hold the rmsd and max_error matrices
rmsd_dict = Dict{String,Matrix{Float64}}()
max_error_dict = Dict{String,Matrix{Float64}}()

for connection in eachindex(connections)
    rmsd_matrix = zeros(length(sel_set), length(rate_set))
    max_error_matrix = zeros(length(sel_set), length(rate_set))

    for (idx, i) in enumerate(sel_set)
        for (idx2, j) in enumerate(rate_set)
            localalpha_params = deepcopy(alpha_params)
            localalpha_params[connection] = (alpha_params[connection][1], j, i)
            localrmsd, localmax_error = loss_function(localalpha_params)
            rmsd_matrix[idx, idx2] = localrmsd
            max_error_matrix[idx, idx2] = localmax_error
        end
    end

    rmsd_dict[connections[connection]] = rmsd_matrix
    max_error_dict[connections[connection]] = max_error_matrix
end

xticks = (1:2:length(rate_set), Int.(round.(rate_set[1:2:end] ./ 1000)))
yticks = (1:2:length(sel_set), sel_set[1:2:end])

for con in connections
    heatmap(limitto.(max_error_dict[con], 2.5), xlabel="rate / 1000", ylabel="selectivity", xticks=xticks, yticks=yticks, title="Max RMSD of fit for $con")
    savefig("alpha_max_error_alldata_cutoff25_$con.png")
end

for con in connections
    heatmap(limitto.(rmsd_dict[con], 2), xlabel="rate / 1000", ylabel="selectivity", xticks=xticks, yticks=yticks, title="Overall RMSD of fit for $con")
    savefig("alpha_overallerror_alldata_cutoff2_$con.png")
end