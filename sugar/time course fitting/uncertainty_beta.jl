include("../../velocity/velocity.jl")

using DataFrames
using CSV
using LinearAlgebra
using Distributions
using Plots

sugars = ["Glc", "Man", "All", "Gal", "Alt", "Tal", "Gul", "Ido"]
sugars = sort(sugars)

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
connections = [x[1] for x in beta_params]

sel_set = [-20:-2; 1:20]
rate_set = 1000:500:20000


# import and process training data
fit_data = CSV.read("expt_data/beta_trainset.csv", DataFrame)
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
            localbeta_params = deepcopy(beta_params)
            localbeta_params[connection] = (beta_params[connection][1], j, i)
            localrmsd, localmax_error = loss_function(localbeta_params)
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
    savefig("beta_max_error_alldata_cutoff25_$con.png")
end

for con in connections
    heatmap(limitto.(rmsd_dict[con], 2), xlabel="rate / 1000", ylabel="selectivity", xticks=xticks, yticks=yticks, title="Overall RMSD of fit for $con")
    savefig("beta_overallerror_alldata_cutoff2_$con.png")
end