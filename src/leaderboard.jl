using TOML
using DataFrames

"""
    find_run_dirs(root_dir::String) -> Vector{String}

Recursively find all directories under `root_dir` that contain a `summary.toml` file.
"""
function find_run_dirs(root_dir::String)
    run_dirs = String[]
    for (dirpath, _, filenames) in walkdir(root_dir)
        if "summary.toml" in filenames
            push!(run_dirs, dirpath)
        end
    end
    return run_dirs
end

"""
    load_leaderboard(root_dirs; quantities=nothing) -> DataFrame

Load all `summary.toml` files found under `root_dirs` (a `String` or `Vector{String}`)
into a DataFrame with one row per run.

- `quantities`: filter to runs whose `out_quantities` overlaps this list (e.g. `["surge"]`).

# Known issue
Per-station stats CSV files (`stats_testing.csv`) use inconsistent column schemas across
model families (hatyan-style vs wave-style). Station-level breakdown is therefore not yet
included in the leaderboard; only `summary.toml` fields are used.
"""
function load_leaderboard(root_dirs; quantities=nothing)
    dirs = root_dirs isa String ? [root_dirs] : collect(root_dirs)
    rows = Dict{String,Any}[]
    for root_dir in dirs
        for run_dir in find_run_dirs(root_dir)
            summary = TOML.parsefile(joinpath(run_dir, "summary.toml"))
            if quantities !== nothing
                run_quantities = get(summary, "out_quantities", String[])
                isempty(intersect(quantities, run_quantities)) && continue
            end
            row = Dict{String,Any}(summary)
            row["run_dir"] = run_dir
            row["out_quantities"] = join(get(summary, "out_quantities", String[]), ", ")
            push!(rows, row)
        end
    end

    isempty(rows) && return DataFrame()

    all_keys = unique(reduce(vcat, collect.(keys.(rows))))
    priority = ["runid", "model_name", "out_quantities", "rmse_testing",
                "n_params", "train_time_s", "description", "run_dir"]
    ordered_keys = [k for k in priority if k in all_keys]
    append!(ordered_keys, sort(filter(k -> k ∉ priority, all_keys)))

    df = DataFrame()
    for k in ordered_keys
        df[!, k] = [get(row, k, missing) for row in rows]
    end
    return df
end

"""
    sort_leaderboard(df::DataFrame, by::String="rmse_testing"; ascending::Bool=true) -> DataFrame

Return `df` sorted by `by`. Rows with `missing` in that column are placed last.
"""
function sort_leaderboard(df::DataFrame, by::String="rmse_testing"; ascending::Bool=true)
    return sort(df, by, rev=!ascending)
end
