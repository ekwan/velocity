using CSV, DataFrames

sugars = ["All", "Alt", "Gal", "Glc", "Gul", "Tal", "Ido", "Man"]

function samesugar(header::String)
    s1,s2 = split(header, "_")
    return s1 == s2
end

df = CSV.read("transient-data/transient_nobias_nocorrelation.csv", DataFrame)

# cols that do not contain "Catalyst"
# cols that are not a sugar to itself
cols = filter(x -> !occursin("Catalyst", x), names(df))
cols = filter(x -> !samesugar(x), cols)
df = df[:, cols]

# lets get the maximum for making each sugar per simulation
function is_traget_sugar(header::String, target::String)
    s1,s2 = split(header, "_")
    return s2 == target
end

sugar_target_maxs = Dict{String, Vector{Float64}}()
for sugar in sugars
    targcols = filter(x -> is_traget_sugar(x, sugar), names(df))
    sugar_target_maxs[sugar] = maximum.(eachrow(df[:, targcols]))
end
df_maxs = DataFrame(sugar_target_maxs) ./ 0.2
numsofsols = sum.(eachrow(df_maxs .> 0.5))
for numsol in [ 1, 2, 3]
    println("Number of solutions with at least $numsol sugar(s) above 0.5: ", sum(numsofsols .>= numsol)/10000)
end

# now by edit distances
# first calculate distances in sugar network
using Graphs
CONNECTIONS = [
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
sugars = sort(["All", "Alt", "Gal", "Glc", "Gul", "Tal", "Ido", "Man"])
sugar_to_index = Dict(sugars .=> 1:length(sugars))
g = SimpleGraph(length(sugars))
for connection in CONNECTIONS
    sugar1, sugar2 = split(connection, "_")
    add_edge!(g, sugar_to_index[sugar1], sugar_to_index[sugar2])
end
shortest_paths = floyd_warshall_shortest_paths(g)
function limit_cols_by_edit_distance(df, threshold=1)
    cols = names(df)
    cols = filter(x -> !occursin("Catalyst", x), cols)
    cols = filter(x -> !samesugar(x), cols)
    cols = filter(x -> shortest_paths.dists[sugar_to_index[get_start_sugar(x)], sugar_to_index[get_target_sugar(x)]] == threshold, cols)
    return df[!, cols]
end
function get_target_sugar(header::String)
    s1,s2 = split(header, "_")
    return s2
end
function get_start_sugar(header::String)
    s1,s2 = split(header, "_")
    return s1    
end
distance_dfs = Vector{DataFrame}(undef, 3)
for i in 1:3
    distance_dfs[i] = limit_cols_by_edit_distance(df, i) ./ 0.2
end

for i in 1:3
    println("Number of solutions with at least 1 sugar above 0.5 and edit distance $i: ", sum(maximum.(eachrow(distance_dfs[i])) .> 0.5) / 10_000)
end

using Plots
histogram([maximum(x) for x in eachrow(distance_dfs[1])], alpha=0.5, bins = 0:0.05:1, label="1 edit")
histogram!([maximum(x) for x in eachrow(distance_dfs[2])],  alpha=0.5, bins = 0:0.05:1, label="2 edits")
histogram!([maximum(x) for x in eachrow(distance_dfs[3])], alpha=0.5, bins = 0:0.05:1, label="3 edits")
xlabel!("Maximum selectivity")
ylabel!("Frequency")
savefig("edit distances.png")

