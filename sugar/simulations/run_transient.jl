include("../../velocity/velocity.jl")

using DataFrames
using CSV
using LinearAlgebra
using Distributions

using Base.Threads

function simulate_transient(network::RxnNetwork)
    """simulate maximum transient concentration of each species in network

    Args:
        network (RxnNetwork): network to simulate

    Returns:
        Array{Float64}: maximum transient concentration of each species for each starting isomer
    """

    tspan = (0.0, 50_000.0)

    # teval is log spaced between 0 and 50000
    teval = (10 .^ (range(2, stop=log10(tspan[2]), length=1_000))...,)

    # alphabetical list of species
    species_ids = sort(collect([x.id for x in network.species]))

    results = []

    # for each start isomer run a simulation
    for s in species_ids
        if (s == "Catalyst") || (s == "Catalyst_Dead")
            continue
        end
        init_conc = Dict(Species(id=s) => 0.2, Species(id="Catalyst") => 0.004)

        run = simulate_timecourse(network, init_conc, tspan, teval).u

        # get the maximum conentration of each species along the simulated simulated timecourse + push results
        max_cons = maximum(hcat(run...), dims=2)
        push!(results, max_cons)
    end

    # combine and return results
    return vcat(results...)
end

function add_correlation!(cor_matrix, i, j, correlation)
    """add correlation to cor_matrix at position i,j + j,i modifying in place

    Args:
        cor_matrix (Matrix{Float64}): correlation matrix
        i (Int): index of first connection
        j (Int): index of second connection
        correlation (Float64): correlation value
    """

    cor_matrix[i, j] = correlation
    cor_matrix[j, i] = correlation
    return cor_matrix
end

function get_species_from_connections(connections)
    """get a list of unique species from a list of connections

    Args:
        connections (Vector{String}): list of connections

    Returns:
        Vector{String}: list of species sorted alphabetically
    """
    species = []
    for connection in connections
        reactant, product = split(connection, "_")
        push!(species, reactant)
        push!(species, product)
    end
    return sort(unique(species))
end

function add_connection_correlation!(cor_matrix::Matrix{Float64}, connections::Vector{String}, cor_cons::Vector{Union{String,Tuple{String,Bool}}}, cor_type::String, correlation::Float64)
    """Given a list of connections, add corresponding correlation to cor_matrix to correlate to each other, modifying in place

    Args:
        cor_matrix (Matrix{Float64}): correlation matrix
        connections (Vector{String}): list of connections, each connection is a string in the form of "A_B" where A is a reactant and B is a product
        cor_cons (Vector{Union{String,Tuple{String,Bool}}}): list of connections to correlate, String is a connection, Tuple{String,Bool} is a connection and a boolean indicating if it is negatie correlation
        cor_type (String): type of correlation, either "rate" or "selectivity"
        correlation (Float64): correlation value, between 0 and 1
    """
    @assert cor_type in ["rate", "selectivity"]

    # loop through all combinations of cor_cons
    for (idx, cor_con1) in enumerate(cor_cons)
        # set direction to be positive
        dir1 = 1

        # if cor_con1 is a string then we need to convert it to a tuple
        # assume positive correlation
        if typeof(cor_con1) == String
            cor_con1 = (cor_con1, false)
        end

        # find first indice of cor_con1 in connection
        cor_con1_idx = findfirst(item -> item == cor_con1[1], connections)

        # first half of the matrix is for rate correlations, second half is for selectivity
        # if cor_type is selectivity then we need to add the length of connections to the index
        if cor_type == "selectivity"
            cor_con1_idx += length(connections)
        end

        # if cor_con1 is negatively correlated to set then we need to change the direction
        if cor_con1[2]
            dir1 = -1
        end

        for cor_con2 in cor_cons[idx+1:end]
            # similarly handle directionality
            dir2 = 1
            if typeof(cor_con2) == String
                cor_con2 = (cor_con2, false)
            end
            cor_con2_idx = findfirst(item -> item == cor_con2[1], connections)
            if cor_type == "selectivity"
                cor_con2_idx += length(connections)
            end
            if cor_con2[2]
                dir2 = -1
            end

            # add correlations
            # fixing directionality by multiplying dir1 and dir2
            add_correlation!(cor_matrix, cor_con1_idx, cor_con2_idx, correlation * dir1 * dir2)
        end
    end

    return cor_matrix
end

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

# correlation ids used for file names
# match order of correlated_connections below
cor_ids = [
    "basin",
    "c3",
    "parallel",
    "twobasin-uc",
    "twobasin-inv",
    "total"
]

# list of correlation scenarios to test
# each scenario is a list of correlation sets
# each correlation set is a list of connections to correlate
# each connection is a string in the form of "A_B" where A is a reactant and B is a product
# if a connection is a tuple then the second element is a boolean indicating if the correlation is negative
# this assumes that connections do not appear in more than one correlation set i.e. there is no cross correlation to handle
correlated_connections = [
    # one basin towards Allose
    [[
        "Glc_All",
        "Alt_All",
        "Gul_All"
    ]],

    # All C3 epimerizations
    [[
        "Glc_All",
        ("Ido_Tal", true),
        "Man_Alt",
        ("Gul_Gal", true),
    ]],

    # parallel parallel edge_cors
    # i.e. each site
    [
        # C3 sites
        [
            "Glc_All",
            ("Ido_Tal", true),
            "Man_Alt",
            ("Gul_Gal", true),
        ],

        # C2 sites
        [
            "Alt_All",
            ("Glc_Man", true),
            "Ido_Gul",
            ("Gal_Tal", true),
        ],

        # C4 sites
        [
            "Gul_All",
            ("Glc_Gal", true),
            "Ido_Alt",
            ("Man_Tal", true),
        ]
    ],

    # two opposite basins with no correlation between them
    [
        # towards Allose
        [
            "Glc_All",
            "Alt_All",
            "Gul_All"
        ],

        # towards Talose
        [
            "Man_Tal",
            "Ido_Tal",
            "Gal_Tal",
        ]
    ],

    # two opposite basins inverse correlation between them
    [[
        "Glc_All",
        "Alt_All",
        "Gul_All", "Man_Tal",
        "Ido_Tal",
        "Gal_Tal",
    ]],

    # total correlation pointing towards Allose
    [[
        "Glc_All",
        "Alt_All",
        "Gul_All",
        ("Glc_Man", true),
        ("Glc_Gal", true),
        "Man_Alt",
        ("Gul_Gal", true),
        "Ido_Alt",
        "Ido_Gul",
        ("Man_Tal", true),
        ("Gal_Tal", true),
        ("Ido_Tal", true),
    ]],
]

# setup correlations to test accross
# include 0 for each (i.e. no correlation)
edge_cors = Float64.(collect(0:0.2:0.8))

# base rate is arbitrarily set to 3000 uM/s
# this only signals approach to equilibrium
base_rate_mu_star = 3000
base_rate_mu = log(base_rate_mu_star)
base_rate_means = fill(base_rate_mu, length(connections))

# selectivity is set to 1 (i.e. no selectivity)
selectivity_mu_star = 1
selectivity_mu = log(selectivity_mu_star)
selectivity_means = fill(selectivity_mu, length(connections))

# combine means into one vector
means = vcat(base_rate_means, selectivity_means)

# number of runs to simulate
number_runs = 10_000

# settings setups
# stdev_params is a list of standard deviation parameters to test in kcal/mol
# correlation_types is a list of correlation types to test
# bias_types is a list of bias types to test
stdev_params = [1.25]
correlation_types = ["rate", "selectivity", "both"]
bias_types = ["none", "rate", "selectivity", "both"]

# in biased simulations correlated connections will have different means
# shift mean by factor of 2.3 (i.e. by 0.5 kcal/mol)
corcon_rate_mean = log(base_rate_mu_star * 2.3)
corcon_selectivity_mean = log(selectivity_mu_star * 2.3)

for (correlated_connection_set, id) in zip(correlated_connections, cor_ids)
    for rate_std in stdev_params

        # setup base rate variance
        base_rate_sigma_kcal = rate_std
        base_rate_sigma = base_rate_sigma_kcal / (1.9872036e-3 * 298.15) # divide by RT
        base_rate_variance = base_rate_sigma^2
        base_rate_variances = fill(base_rate_variance, length(connections))

        for sel_std in stdev_params

            # setup selectivity variance
            selectivity_sigma_kcal = sel_std
            selectivity_sigma = selectivity_sigma_kcal / (1.9872036e-3 * 298.15) # divide by RT
            selectivity_variance = selectivity_sigma^2
            selectivity_variances = fill(selectivity_variance, length(connections))

            # combine variances into one vector
            variances = vcat(base_rate_variances, selectivity_variances)

            for cor_type in correlation_types
                for bias_type in bias_types

                    # copy means to local_means to adjust correlated connection means if needed
                    local_means = copy(means)

                    if bias_type != "none"
                        for correlated_cons in correlated_connection_set
                            for connection in correlated_cons

                                # for each connection get connection and direction of correlation                                
                                if typeof(connection) == Tuple{String,Bool}
                                    con = connection[1]
                                    dir = connection[2]
                                else
                                    con = connection
                                    dir = false
                                end

                                # figure out which means to modify
                                index = findfirst(x -> x == con, connections)

                                # rate doesn't have a direction
                                if (bias_type == "rate") || (bias_type == "both")
                                    local_means[index] = corcon_rate_mean
                                end

                                # for selectivity mean shift move in the right direction
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

                    # now we have all the parameters we need to run the simulation
                    # run the simulation for each edge_cor in edge_cors
                    for edge_cor in edge_cors

                        # file title to be programatically set based on parameters
                        title = id * "_" * string(rate_std) * "_" * string(sel_std) * "_" * string(cor_type) * "_" * string(bias_type) * "_" * string(edge_cor) * ".csv"

                        try # catch errors for correlation matrix issues gracefully

                            # initial correlation matrix is just square identity matrix
                            cor = Matrix{Float64}(I, length(connections) * 2, length(connections) * 2)

                            for correlated_cons in correlated_connection_set

                                # add rate correlation
                                if (cor_type == "rate") || (cor_type == "both")
                                    add_connection_correlation!(cor, connections, Vector{Union{String,Tuple{String,Bool}}}(correlated_cons), "rate", edge_cor)
                                end

                                # add selectivity correlation
                                if (cor_type == "selectivity") || (cor_type == "both")
                                    add_connection_correlation!(cor, connections, Vector{Union{String,Tuple{String,Bool}}}(correlated_cons), "selectivity", edge_cor)
                                end


                            end

                            # convert corerlation to covariance by multiplying by variances
                            cov_matrix = cor .* variances

                            # create multivariate log normal distribution
                            d = MvLogNormal(local_means, cov_matrix)

                            # setup for parallel runs
                            # vectors to store results and setups, one for each run
                            setups = Vector(undef, number_runs)
                            results = Vector(undef, number_runs)

                            # run simulations in parallel
                            Threads.@threads for i in 1:number_runs
                                # get a random sample from the distribution
                                x = rand(d)
                                setups[i] = x

                                # split the sample into rates and selectivities
                                rates = x[1:length(connections)]
                                sels = x[length(connections)+1:end]

                                # create a network from the parameters
                                params = collect(zip(connections, rates, sels))
                                network = get_network_cat(params)

                                # simulate the network and store the results
                                results[i] = simulate_transient(network)
                            end

                            # combine results and setups into matrix
                            results = permutedims(hcat(results...))
                            cols = sort(vcat(get_species_from_connections(connections), ["Catalyst", "Catalyst_Dead"]))

                            # setup column names
                            full_cols = []
                            for col in cols
                                if (col == "Catalyst") || (col == "Catalyst_Dead")
                                    continue
                                end
                                for subcol in cols
                                    local_col = col * "_" * subcol
                                    push!(full_cols, local_col)
                                end
                            end

                            # save results
                            results = DataFrame(results, full_cols)
                            CSV.write("results_" * title, results)

                            # save setups
                            setups = permutedims(hcat(setups...))
                            col_labels_a = [string(connection) * "_base_rate" for connection in connections]
                            col_labels_s = [string(connection) * "_selectivity" for connection in connections]
                            col_labels = vcat(col_labels_a, col_labels_s)
                            setups = DataFrame(setups, col_labels)
                            CSV.write("setups_" * title, setups)

                        catch e
                            println(title * " failed with error: " * string(e))
                        end
                    end
                end
            end
        end
    end
end