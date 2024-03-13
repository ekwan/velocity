using Parameters
using DifferentialEquations
import Base.isless

"""
    struct Species

A struct representing a species in a reaction network.

# Fields
- `id::String`: The unique identifier for the species.
- `description::String`: An optional description of the species.

"""
@with_kw struct Species
    id::String
    description::String = ""
end
isless(a::Species, b::Species) = isless(a.id, b.id)

abstract type AbstractReaction end

"""
    struct Reaction <: AbstractReaction

A struct representing an elementary chemical reaction.

# Fields
- `reactants::Dict{Species,Int64}`: A dictionary of species and their stoichiometries in the reactants.
- `products::Dict{Species,Int64}`: A dictionary of species and their stoichiometries in the products.
- `catalysts::Dict{Species,Int64}`: A dictionary of species and their stoichiometries acting as catalysts. Default is an empty dictionary.
- `rate_constant::Tuple{Float64,Float64}`: A tuple of forward and reverse rate constants.
- `description::String`: A description of the reaction. Default is an empty string.
- `id::String`: An identifier for the reaction.

"""
@with_kw struct Reaction <: AbstractReaction
    reactants::Dict{Species,Int64}
    products::Dict{Species,Int64}
    catalysts::Dict{Species,Int64} = Dict{Species,Int64}()
    rate_constant::Tuple{Float64,Float64}
    description::String = ""
    id::String
end

"""
    CompositeReaction

A struct representing a composite reaction, i.e. where order must be specified separately.

# Fields
- `reactants::Dict{Species,Int64}`: A dictionary of species and their stoichiometries as reactants.
- `products::Dict{Species,Int64}`: A dictionary of species and their stoichiometries as products.
- `catalysts::Dict{Species,Int64}`: A dictionary of species and their stoichiometries as catalysts. (default: `Dict{}`)
- `order::Dict{Species,Int64}`: A dictionary specifying the order of species in the reaction.
- `rate_constant::Tuple{Float64,Float64}`: A tuple of forward and reverse rate constants.
- `description::String`: A description of the reaction. (default: "")
- `id::String`: An identifier for the reaction.

"""
@with_kw struct CompositeReaction <: AbstractReaction
    reactants::Dict{Species,Int64}
    products::Dict{Species,Int64}
    catalysts::Dict{Species,Int64} = Dict{}
    order::Dict{Species,Int64}
    rate_constant::Tuple{Float64,Float64}
    description::String = ""
    id::String
end

"""
    struct RxnNetwork

A struct representing a reaction network.

# Fields
- `reactions`: A tuple of reactions or composite reactions.
- `species`: A tuple of species.

"""
struct RxnNetwork
    reactions::Tuple{AbstractReaction,Vararg{AbstractReaction}}
    species::Tuple{Species,Vararg{Species}}
end

"""
    get_id(s)

Get the ID of an object.

# Arguments
- `s`: The object to get the ID from.

# Returns
The ID of the object.
"""
function get_id(s)
    return s.id
end

"""
    get_species(reactions::Vector{AbstractReaction})

Given a list of reactions, this function returns a list of unique species involved in those reactions.

# Arguments
- `reactions`: A vector of `AbstractReaction` objects representing the reactions.

# Returns
A tuple containing the unique species involved in the reactions.

"""
function get_species(reactions::Vector{AbstractReaction})
    # given list of reactions, return list of unique species
    species = Species[]  # Use an array to collect species
    for rxn in reactions
        for s in keys(rxn.reactants)
            push!(species, s)
        end
        for s in keys(rxn.products)
            push!(species, s)
        end
        for s in keys(rxn.catalysts)
            push!(species, s)
        end
    end

    unique_species = unique(species)
    return (unique_species...,)  # Convert the array to a tuple
end

"""
    get_rate_vector(concentrations::Dict{Species,Float64}, network::RxnNetwork)

Given a dictionary of species concentrations and a reaction network, this function calculates the rate vector.
The rate vector represents the rate of change of each species in the network.

# Arguments
- `concentrations`: A dictionary mapping species to their concentrations.
- `network`: A reaction network object.

# Returns
- `rate_vector`: A dictionary mapping species to their rate of change.

"""
function get_rate_vector(concentrations::Dict{Species,Float64}, network::RxnNetwork)
    # given a network, return the stoichiometry matrix
    species = network.species

    rate_vector = Dict{Species,Float64}()
    for s in species
        rate_vector[s] = 0.0
    end

    for rxn in network.reactions
        rate_f = rxn.rate_constant[1]
        rate_b = rxn.rate_constant[2]
        # if rxn is CompositeReaction then we use order instead of stoich
        if typeof(rxn) == CompositeReaction
            orders = rxn.order

            for (s, o) in orders
                rate_f *= concentrations[s]^o
            end
            for (s, o) in orders
                rate_b *= concentrations[s]^o
            end
        else # just use the stoichiometries
            for (s, o) in rxn.reactants
                rate_f *= concentrations[s]^o
            end
            for (s, o) in rxn.products
                rate_b *= concentrations[s]^o
            end

            for (s, o) in rxn.catalysts
                rate_f *= concentrations[s]^o
                rate_b *= concentrations[s]^o
            end
        end

        rate = rate_f - rate_b

        for s in species
            if haskey(rxn.reactants, s)
                rate_vector[s] -= rate
            end
            if haskey(rxn.products, s)
                rate_vector[s] += rate
            end
        end
    end

    return rate_vector
end

function simulate_timecourse(network::RxnNetwork, initial_conc::Dict{Species,Float64}, tspan::Tuple{Float64,Float64}, t_eval::Tuple{Float64,Vararg{Float64}})
    # define ODE
    # Convert initial_conc to an Array
    species_order = sort(collect(keys(initial_conc)))  # to ensure consistent order
    initial_conc_array = [initial_conc[species] for species in species_order]

    # Modify the f! function0))
    function f!(du, u, p, t)
        # Convert u back to a Dict
        u_dict = Dict(species_order[i] => u[i] for i in 1:length(u))

        # Get the rate vector
        rate_vector = get_rate_vector(u_dict, network)

        # Calculate the derivative
        for i in 1:length(du)
            du[i] = rate_vector[species_order[i]]
        end
    end

    # Define the problem
    prob = ODEProblem(f!, initial_conc_array, tspan)

    # Solve the problem
    sol = solve(prob, saveat=t_eval)

    return sol
end

"""
    get_rate_constants(base_rate::Float64, selectivity::Float64)

Given a base rate and selectivity, return a tuple of forward and reverse rate constants.

# Arguments
- `base_rate::Float64`: The base rate.
- `selectivity::Float64`: The selectivity.

# Returns
A tuple of forward and reverse rate constants.

"""
function get_rate_constants(base_rate::Float64, selectivity::Float64)
    # forward rate is base_rate * sqrt(selectivity)
    # reverse rate is base_rate / sqrt(selectivity)
    return (base_rate * sqrt(selectivity) * 1e-6, base_rate / sqrt(selectivity) * 1e-6)
end

"""
    back_convert_sel(sel::Float64)

Converts a selectivity into a ratio (i.e. 0.5 -> -2)

# Arguments
- `sel::Float64`: The selectivity value to be converted.

# Returns
- The selectivity ratio.

"""
function back_convert_sel(sel::Float64)
    if sel < 0
        return -1 / sel
    else
        return sel
    end
end

"""
    covert_sel(sel)

Converts a ratio selectivity into a selectivity (i.e. -2 -> 0.5)

# Arguments
- `sel`: The selectivity ratio to be converted.

# Returns
- The selectivity.

"""
function convert_sel(sel)
    if sel < 1
        return -1 / sel
    else
        return sel
    end
end