using Parameters
using DifferentialEquations
import Base.isless

@with_kw struct Species
    """ This represents a single chemical species.

    Fields:
        id (String): abbreviation
        description (String): an optional longer name
    """
    id::String
    description::String = ""
end

# extend isless to compare species
isless(a::Species, b::Species) = isless(a.id, b.id)

# AbstractReaction is either a Reaction or CompositeReaction
abstract type AbstractReaction end

@with_kw struct Reaction <: AbstractReaction
    """ This represents an elementary chemical reaction and is a type of AbstractReaction

    Fields:
        reactants (Dict{Species,Int64}): A dictionary of species and their stoichiometries in the reactants.
        products (Dict{Species,Int64}): A dictionary of species and their stoichiometries in the products.
        catalysts (Dict{Species,Int64}): An optional dictionary of species and their stoichiometries acting as catalysts.
        rate_constant (Tuple{Float64,Float64}): A tuple of forward and reverse rate constants.
        description (String): An optional description of the reaction. Default is an empty string.
        id (String): An identifier for the reaction.

    """

    reactants::Dict{Species,Int64}
    products::Dict{Species,Int64}
    catalysts::Dict{Species,Int64} = Dict{Species,Int64}()
    rate_constant::Tuple{Float64,Float64}
    description::String = ""
    id::String
end

@with_kw struct CompositeReaction <: AbstractReaction
    """This represents a composite reaction, i.e. where order must be specified separately instead of using stoichiometries.

    Fields:
        reactants (Dict{Species,Int64}): A dictionary of species and their stoichiometries as reactants.
        products (Dict{Species,Int64}): A dictionary of species and their stoichiometries as products.
        catalysts (Dict{Species,Int64}): An optional dictionary of species and their stoichiometries acting as catalysts.
        order (Tuple{Dict{Species,Int64},Dict{Species,Int64}}): A tuple of dictionaries specifying the order of species in the forward and reverse reactions.
        rate_constant (Tuple{Float64,Float64}): A tuple of forward and reverse rate constants.
        description (String): An optional description of the reaction. Default is an empty string.
        id (String): An identifier for the reaction.

    """

    reactants::Dict{Species,Int64}
    products::Dict{Species,Int64}
    catalysts::Dict{Species,Int64} = Dict{}
    order::Tuple{Dict{Species,Int64},Dict{Species,Int64}}
    rate_constant::Tuple{Float64,Float64}
    description::String = ""
    id::String
end

struct RxnNetwork
    """This represents a reaction network.

    Fields:
        reactions (Tuple{AbstractReaction,Vararg{AbstractReaction}}): A tuple of reactions or composite reactions.
        species (Tuple{Species,Vararg{Species}}): A tuple of species.
    """
    reactions::Tuple{AbstractReaction,Vararg{AbstractReaction}}
    species::Tuple{Species,Vararg{Species}}
end

function get_species(reactions::Vector{AbstractReaction})
    """Get the unique species involved in a list of reactions.

    Args:
        reactions (Vector{AbstractReaction}): A list of reactions.

    Returns:
        unique_species (Tuple{Species}): A tuple of unique species.
    """

    species = Species[]

    # push all species into the species array
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

    # get unique species
    species = sort(unique(species))

    # cast to tuple and return
    return (unique_species...,)  # Convert the array to a tuple
end

function get_rate_vector(concentrations::Vector{Float64}, network::RxnNetwork)
    """ Get the rate vector for a given network and concentrations. This is the rate of change of each species in the network.

    Args:
        concentrations (Vector{Float64}): A vector of concentrations, order must match network.species.
        network (RxnNetwork): A reaction network.

    Returns:
        rate_vector (Vector{Float64}): A vector of species and their rate of change.
    """

    # get species
    species = network.species

    # generate an zero rate vector
    rate_vector = zeros(Float64, length(species))

    # loop through each reaction and determine its effect on rate_vector
    for rxn in network.reactions

        # get rate constants
        rate_f = rxn.rate_constant[1] # forward rate
        rate_b = rxn.rate_constant[2] # back rate

        if typeof(rxn) == CompositeReaction # use orders for exponents
            # get order dictionaries
            order_f, order_b = rxn.order

            for s_index in eachindex(species)
                s = species[s_index]

                # if the forward reaction is ordered with respect to this species
                # then multiply the rate by the concentration to the power of the order
                if haskey(order_f, s)
                    rate_f *= concentrations[s_index]^order_f[s]
                end

                # if the back reaction is ordered with respect to this species
                # then multiply the rate by the concentration to the power of the order
                if haskey(order_b, s)
                    rate_b *= concentrations[s_index]^order_b[s]
                end
            end

        else # just use the stoichiometries

            for s_index in eachindex(species)
                s = species[s_index]

                # if the forward reaction is ordered with respect to this species
                # then multiply the rate by the concentration to the power of the order
                if haskey(rxn.reactants, s)
                    rate_f *= concentrations[s_index]^rxn.reactants[s]
                end

                # if the back reaction is ordered with respect to this species
                # then multiply the rate by the concentration to the power of the order
                if haskey(rxn.products, s)
                    rate_b *= concentrations[s_index]^rxn.products[s]
                end

                # if the reaction is catalyzed by this species
                # then multiply the rate by the concentration to the power of the order
                if haskey(rxn.catalysts, s)
                    rate_f *= concentrations[s_index]^rxn.catalysts[s]
                    rate_b *= concentrations[s_index]^rxn.catalysts[s]
                end
            end
        end

        rate = rate_f - rate_b

        # calculate change in each species
        for s_index in eachindex(species)
            s = species[s_index]

            if haskey(rxn.reactants, s)
                rate_vector[s_index] -= rate
            end
            if haskey(rxn.products, s)
                rate_vector[s_index] += rate
            end
        end
    end

    return rate_vector
end

function simulate_timecourse(network::RxnNetwork, initial_conc::Dict{Species,Float64}, tspan::Tuple{Float64,Float64}, t_eval::Tuple{Float64,Vararg{Float64}})
    """Given a network and initial concentrations and time properties calculate timepoints by numeric integration.

    Args:
        network (RxnNetwork): A reaction network.
        initial_conc (Dict{Species,Float64}): A dictionary of initial concentrations.
        tspan (Tuple{Float64,Float64}): tuple of time span to integrate
        t_eval (Tuple{Float64,Vararg{Float64}}): tuple of time points to keep track of

    Returns:
        sol (Differential equation solution): sol.u = list of concentration vectors at each time point, sol.t = timepoints
    """
    # convert initial_conc to an array that matches the order of species in network
    # if a species is not in initial_conc, then it is assumed to be 0.0
    initial_conc_array = [get(initial_conc, s, 0.0) for s in network.species]


    # f! function
    function f!(du, u, p, t)
        # get rate vector and modify du in place
        du .= get_rate_vector(u, network)
    end

    # Define the problem
    prob = ODEProblem(f!, initial_conc_array, tspan)

    # Solve the problem
    sol = solve(prob, saveat=t_eval)

    return sol
end

function get_rate_constants(base_rate::Float64, selectivity::Float64)
    """ given a base rate and selectivity, return a tuple of forward and reverse rate constants.

    Args:
        base_rate (Float64): The base rate in uM/s.
        selectivity (Float64): The selectivity.

    Returns:
        Tuple{Float64,Float64}: A tuple of forward and reverse rate constants in M/s.
    """

    # forward rate is base_rate * sqrt(selectivity)
    # reverse rate is base_rate / sqrt(selectivity)
    return (base_rate * sqrt(selectivity) * 1e-6, base_rate / sqrt(selectivity) * 1e-6)
end

function back_convert_sel(sel::Float64)
    """ Convert a selectivity into a ratio (i.e. 0.5 -> -2)

    Args:
        sel (Float64): The selectivity value to be converted.

    Returns:
        Float64: The selectivity ratio.
    """
    if sel < 0
        return -1 / sel
    else
        return sel
    end
end

function convert_sel(sel)
    """ Convert a selectivity ratio into a selectivity (i.e. -2 -> 0.5)

    Args:
        sel (Float64): The selectivity ratio to be converted.

    Returns:
        Float64: The selectivity.
    """
    if sel < 1
        return -1 / sel
    else
        return sel
    end
end
