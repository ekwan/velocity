# import Pkg; Pkg.add(["DifferentialEquations", "Parameters", "Distributions", "LinearAlgebra", "DataFrames", "CSV"])
include("../../velocity/velocity.jl")

using DataFrames
using CSV
using LinearAlgebra
using Distributions

using Base.Threads

function simulate_asymptote(network::RxnNetwork)
    tspan = (0.0, 50000.0)

    # glucose and idose are opposite to each other
    init_conc1 = get_initial_conc(network.species,[(Species(id="Glc"), 0.2),(Species(id="Catalyst"), 0.004)])
    init_conc2 = get_initial_conc(network.species,[(Species(id="Ido"), 0.2),(Species(id="Catalyst"), 0.004)])

    # run with init_conc1 and get final concentration list
    run1 = simulate_timecourse(network, init_conc1, tspan, tspan).u[end]

    # run with init_conc2 and get final concentration list
    run2 = simulate_timecourse(network, init_conc2, tspan, tspan).u[end]

    # check if the final concentrations are approximately the same
    if isapprox(run1, run2, atol=0.01)
        return run1
    else
        while !isapprox(run1, run2, atol=0.01)
            tspan = (0.0, tspan[2]*2)
            run1 = simulate_timecourse(network, init_conc1, tspan, tspan).u[end]
            run2 = simulate_timecourse(network, init_conc2, tspan, tspan).u[end]
        end
        return run1
    end
end

function add_correlation!(cor_matrix, i, j, correlation)
    # add correlation to cor_matrix at position i,j modifying in place
    cor_matrix[i,j] = correlation
    cor_matrix[j,i] = correlation
    return cor_matrix
end

function get_species_from_connections(connections)
    species = []
    for connection in connections
        reactant, product = split(connection, "_")
        push!(species, reactant)
        push!(species, product)
    end
    return unique(species)
end

function add_connection_correlation!(cor_matrix::Matrix{Float64}, connections::Vector{String}, cor_cons::Vector{Union{String, Tuple{String, Bool}}}, cor_type::String, correlation::Float64)
    # add correlation to cor_matrix at position i,j without modifying in place

    @assert cor_type in ["rate", "selectivity"]

    for (idx, cor_con1) in enumerate(cor_cons)
        dir1 = 1
        
        if typeof(cor_con1) == String
            cor_con1 = (cor_con1, false)
        end

        # find first indice of cor_con1 in connection
        cor_con1_idx = findfirst(item -> item == cor_con1[1], connections)

        if cor_type == "selectivity"
            cor_con1_idx += length(connections)
        end

        # if cor_con1 is a reverse reaction then we need to change the direction
        if cor_con1[2]
            dir1 = -1
        end

        for cor_con2 in cor_cons[idx+1:end]
            dir2 = 1

            if typeof(cor_con2) == String
                cor_con2 = (cor_con2, false)
            end

            # find first indice of cor_con2 in connection
            cor_con2_idx = findfirst(item -> item == cor_con2[1], connections)

            if cor_type == "selectivity"
                cor_con2_idx += length(connections)
            end

            # if cor_con2 is a reverse reaction then we need to change the direction
            if cor_con2[2]
                dir2 = -1
            end

            # add correlations
            add_correlation!(cor_matrix, cor_con1_idx, cor_con2_idx, correlation*dir1*dir2)
        end
    end

    return cor_matrix
end

function get_initial_conc(allspecies::Tuple{Species,Vararg{Species}}, species::Vector{Tuple{Species, Float64}})
    # given a list of species in the form of a list of tuples (species, concentration) where species is a string in the form of "A" where A is a species
    # return a dictionary of species and concentrations

    initial_conc = Dict{Species, Float64}()

    for s in allspecies
        initial_conc[s] = 0.0
    end

    for (s, c) in species
        initial_conc[s] = c
    end
    return initial_conc
end


function get_network_cat(params::Vector{Tuple{String, Float64, Float64}},catalyst_death::Union{Float64,Bool}=false)
    # given a list of simple reactions in the form of a list of tuples (connection, base_rate, selectivity) where connection is a string in the form of "A_B" where A is a rectant and B is a products
    # return a network of reactions

    species = Species[]
    reactions = Reaction[]
    for (connection,base_rate,selectivity) in params
        reactant, product = split(connection, "_")
        push!(species, Species(id=reactant))
        push!(species, Species(id=product))
        push!(reactions, Reaction(reactants = Dict(Species(id = reactant) => 1), products = Dict(Species(id = product) => 1), catalysts = Dict(Species(id = "Catalyst") => 1), rate_constant=get_rate_constants(base_rate, back_convert_sel(selectivity)), id=connection))
    end

    push!(species, Species(id="Catalyst"))
    push!(species, Species(id="Catalyst_Dead"))

    # if catalyst_death is a float then we add a reaction that kills the catalyst
    if typeof(catalyst_death) == Float64
        push!(reactions, Reaction(reactants = Dict(Species(id = "Catalyst") => 1), products = Dict(Species(id = "Catalyst_Dead") => 1), rate_constant=(catalyst_death,0), id="cat_death"))
    end

    species = unique(species)
    return RxnNetwork((reactions...,), (species...,))
end

connections = [
    "Glc_All", 
    "Glc_Man",
    "Glc_Gal",
    "Man_Tal",
    "Man_Alt",
    "Gal_Tal",
    "Alt_All",
    "Gul_Gal",
    "Gul_All", 
    "Ido_Alt",
    "Ido_Gul",
    "Ido_Tal",
]

cor_ids = [
    "basin",
    "c3",
    "parallel",
    "twobasin-uc",
    "twobasin-inv",
    "total"
]

correlated_connections = [
    # one basin
    [[
        "Glc_All",
        "Alt_All",
        "Gul_All"
    ]],

    # c3 reactivity
    [[
        "Glc_All",
        ("Ido_Tal",true),
        "Man_Alt",
        ("Gul_Gal",true),
    ]],

    # parallel site reactivity no correlation between sites
    [[
        "Glc_All",
        ("Ido_Tal",true),
        "Man_Alt",
        ("Gul_Gal",true),
    ],
    [
        "Alt_All", #C2
        ("Glc_Man",true), #C2
         "Ido_Gul", #C2
         ("Gal_Tal",true), #C2
     ],
     [
        "Gul_All", #C4
        ("Glc_Gal",true), #C4
        "Ido_Alt", #C4
        ("Man_Tal",true), #C4
     ]
    ],

    #two basins no correlation
    [[
        "Glc_All",
        "Alt_All",
        "Gul_All"
    ],
    [
        "Man_Tal",
        "Ido_Tal",
        "Gal_Tal",
    ]
    ],

    #two basins inverse correlation
    [[
        "Glc_All",
        "Alt_All",
        "Gul_All",

        "Man_Tal",
        "Ido_Tal",
        "Gal_Tal",
    ]],

    #total
    [[
        "Glc_All",
        "Alt_All",
        "Gul_All",
        ("Glc_Man",true),
        ("Glc_Gal",true),
        "Man_Alt",
        ("Gul_Gal",true),
        "Ido_Alt",
        "Ido_Gul",
        ("Man_Tal",true),
        ("Gal_Tal",true),
        ("Ido_Tal",true),
    ]],
]

edge_cors = Float64.(collect(0:0.2:0.8))


# generate a list of means length(connections)*2
base_rate_mu_star = 3000
base_rate_mu = log(base_rate_mu_star)
base_rate_means = fill(base_rate_mu, length(connections))

selectivity_mu_star = 1
selectivity_mu = log(selectivity_mu_star)
selectivity_means = fill(selectivity_mu, length(connections))

means = vcat(base_rate_means, selectivity_means)

corcon_rate_mean = base_rate_mu
corcon_selectivity_mean = selectivity_mu



number_runs = 10000

# settings setups
# correlation id
# correlations
# std deviation Parameters
# correlation type
# bias

stdev_params = [0.75, 1.25]
correlation_types = ["rate", "selectivity", "both"]
bias_types = ["none", "rate", "selectivity", "both"]
corcon_rate_mean = log(base_rate_mu_star * 2.3)
corcon_selectivity_mean = log(selectivity_mu_star * 2.3)

for (correlated_connection_set, id) in zip(correlated_connections,cor_ids)
    for rate_std in stdev_params
        base_rate_sigma_kcal = rate_std
        base_rate_sigma = base_rate_sigma_kcal / (1.9872036e-3 * 298.15)
        base_rate_variance = base_rate_sigma^2
        base_rate_variances = fill(base_rate_variance, length(connections))

        for sel_std in stdev_params
            selectivity_sigma_kcal = sel_std
            selectivity_sigma = selectivity_sigma_kcal / (1.9872036e-3 * 298.15)
            selectivity_variance = selectivity_sigma^2
            selectivity_variances = fill(selectivity_variance, length(connections))

            variances = vcat(base_rate_variances, selectivity_variances)

            for cor_type in correlation_types
                for bias_type in bias_types
                    local_means = copy(means)
                    
                    if bias_type != "none"
                        for correlated_cons in correlated_connection_set
                            for connection in correlated_cons
                                if typeof(connection) == Tuple{String,Bool}
                                    con = connection[1]
                                    dir = connection[2]
                                else
                                    con = connection
                                    dir = false
                                end

                                index = findfirst(x -> x == con, connections)

                                if (bias_type == "rate") || (bias_type == "both")
                                    local_means[index] = corcon_rate_mean
                                end

                                if (bias_type == "selectivity") || (bias_type == "both")
                                    if dir
                                        local_means[index+length(connections)] = -1 * corcon_selectivity_mean
                                    else
                                        local_means[index+length(connections)] = corcon_selectivity_mean
                                    end
                                end
                            end
                        end
                    end

                    for edge_cor in edge_cors
                        title = id*"_"*string(rate_std)*"_"*string(sel_std)*"_"*string(cor_type)*"_"*string(bias_type)*"_"*string(edge_cor)*".csv"

                        try

                            cor = Matrix{Float64}(I, length(connections)*2, length(connections)*2)
                            
                            for correlated_cons in correlated_connection_set

                                if (cor_type == "rate") || (cor_type == "both")
                                    add_connection_correlation!(cor,connections,Vector{Union{String, Tuple{String, Bool}}}(correlated_cons), "rate", edge_cor)
                                end

                                if (cor_type == "selectivity") || (cor_type == "both")
                                    add_connection_correlation!(cor,connections,Vector{Union{String, Tuple{String, Bool}}}(correlated_cons), "selectivity", edge_cor)
                                end


                            end
                            # insert correlation logic here

                            # insert means/variance logic here

                            cov_matrix = cor .* variances         
                            d = MvLogNormal(local_means, cov_matrix)

                            setups = Vector(undef, number_runs)
                            results = Vector(undef, number_runs)

                            Threads.@threads for i in 1:number_runs
                                x = rand(d)
                                setups[i] = x

                                rates = x[1:length(connections)]
                                sels = x[length(connections)+1:end]

                                params = collect(zip(connections, rates, sels))

                                network = get_network_cat(params)
                                results[i] = simulate_asymptote(network)
                            end

                            results = permutedims(hcat(results...))
                            cols = sort(vcat(get_species_from_connections(connections), ["Catalyst", "Catalyst_Dead"]))
                            results = DataFrame(results, cols)
                            CSV.write("results_"*title, results)

                            setups = permutedims(hcat(setups...))
                            col_labels_a = [string(connection)*"_base_rate" for connection in connections]
                            col_labels_s = [string(connection)*"_selectivity" for connection in connections]
                            col_labels = vcat(col_labels_a, col_labels_s)
                            setups = DataFrame(setups, col_labels)
                            CSV.write("setups_"*title, setups)

                        catch e
                            println(title*" failed with error: "*string(e))
                        end
                    end
                end
            end
        end
    end
end