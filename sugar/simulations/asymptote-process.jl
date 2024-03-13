using CSV
using DataFrames
using Plots

sugars = ["All", "Alt", "Gal", "Glc", "Gul", "Tal", "Ido", "Man"]

ids = ["basin", "c3", "parallel", "total", "twobasin-inv", "twobasin-uc"]
bias_types = ["none", "selectivity"]
edge_cors = [0.0, 0.2, 0.4, 0.6, 0.8]

# 6 dimensional matrix of dataframes
dfs = Array{DataFrame}(undef, length(ids), length(edge_cors), length(bias_types))

#get percent > cutoff for a given sugar
function get_prob_sugar(df, sugar="All", cutoff=0.5)
    @assert sugar in sugars || sugar == "Any"

    if sugar == "Any"
        return sum(maximum.(eachrow(df)) .> cutoff) / length(df[!, "All"])
    else
        return sum(df[!, sugar] .> cutoff) / length(df[!, sugar])
    end
end

sugar_interest = "All"
cutoff = 0.5

# populate the matrix + print results
for (idx1, id) in enumerate(ids)
    for (idx2, edge_cor) in enumerate(edge_cors)
        for (idx3, bias) in enumerate(bias_types)
            title = "results_$(id)_1.25_1.25_selectivity_$(bias)_$(edge_cor).csv"
            try
                dfs[idx1, idx2, idx3] = CSV.read("asymptote-data/" * title, DataFrame)[!, sugars] ./ 0.2 # only sugars + molarity
                println("Read file: id: $id, edge_cor: $edge_cor, bias: $bias")
                println("Probability of 'Allose' > 0.5: ", get_prob_sugar(dfs[idx1, idx2, idx3], sugar_interest, cutoff))
            catch
                println("Error reading file: $title")
            end
        end
    end
end
