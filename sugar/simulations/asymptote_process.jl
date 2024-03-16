using CSV
using DataFrames

sugars = ["All", "Alt", "Gal", "Glc", "Gul", "Tal", "Ido", "Man"]
ids = ["basin", "c3", "parallel", "total", "twobasin-inv", "twobasin-uc"]
bias_types = ["none", "selectivity"]
edge_cors = [0.0, 0.2, 0.4, 0.6, 0.8]

# initialize matrix of dataframes to store results
dfs = Array{DataFrame}(undef, length(ids), length(edge_cors), length(bias_types))

function get_prob_sugar(df, sugar="All", cutoff=0.5)
    """given a dataframe of asymptote results, return the probability of a sugar being greater than a cutoff value

    Args:
        df (DataFrame): dataframe of asymptote results
        sugar (str): sugar to check or "Any" for all sugars
        cutoff (float): cutoff value

    Returns:
        float: probability of sugar being greater than cutoff
    """

    @assert sugar in sugars || sugar == "Any"

    # sum(boolean array) / length(boolean array) is probability of true
    if sugar == "Any"
        return sum(maximum.(eachrow(df)) .> cutoff) / length(df[!, "All"])
    else
        return sum(df[!, sugar] .> cutoff) / length(df[!, sugar])
    end
end

# populate the matrix by reading result files
for (idx1, id) in enumerate(ids)
    for (idx2, edge_cor) in enumerate(edge_cors)
        for (idx3, bias) in enumerate(bias_types)
            title = "results_$(id)_1.25_1.25_selectivity_$(bias)_$(edge_cor).csv"
            try
                dfs[idx1, idx2, idx3] = CSV.read("asymptote-data/" * title, DataFrame)[!, sugars] ./ 0.2 # only sugars + molarity
            catch
                println("Error reading file: $title")
            end
        end
    end
end

sugar_interest = "All"
cutoff = 0.5

# print out results
for (idx1, id) in enumerate(ids)
    for (idx2, edge_cor) in enumerate(edge_cors)
        for (idx3, bias) in enumerate(bias_types)
            println("$(id) $(edge_cor) $(bias): $(get_prob_sugar(dfs[idx1, idx2, idx3], sugar_interest, cutoff))")
        end
    end
end