using Dates
using TOML

"""
    load_data(data_settings::Dict{String,Any}; base_dir::String=".")
        -> Dict{String, NamedTuple}

Load time series from files described by `data_settings` and return a dict
mapping each split label to `(input=..., target=...)`.

`base_dir` is the directory against which relative paths in the settings are
resolved (typically `dirname` of the TOML file).
"""
function load_data(data_settings::Dict{String,Any}; base_dir::String=".")
    haskey(data_settings, "files")    || error("load_data: data_settings missing \"files\".")
    haskey(data_settings, "model_io") || error("load_data: data_settings missing \"model_io\".")
    file_entries = data_settings["files"]
    model_io     = data_settings["model_io"]
    haskey(model_io, "input")  || error("load_data: model_io missing \"input\".")
    haskey(model_io, "target") || error("load_data: model_io missing \"target\".")
    input_vars   = model_io["input"]
    target_vars  = model_io["target"]

    for (i, entry) in enumerate(file_entries)
        _validate_file_entry(entry, i)
    end

    splits = unique(entry["split"] for entry in file_entries)

    result = Dict{String, NamedTuple}()
    for split in splits
        flat = Dict{String, TimeSeries}()
        for entry in filter(e -> e["split"] == split, file_entries)
            _load_entry!(flat, entry, base_dir)
        end

        flat = _intersect_times(flat, split)

        for (role, vars) in (("input", input_vars), ("target", target_vars))
            for k in vars
                haskey(flat, k) || error(
                    "load_data: variable \"$k\" declared in model_io.$role is not " *
                    "provided by any file in split \"$split\".")
            end
        end

        input  = Dict{String, TimeSeries}(k => flat[k] for k in input_vars)
        target = Dict{String, TimeSeries}(k => flat[k] for k in target_vars)
        result[split] = (input=input, target=target)
    end
    return result
end

# Keys permitted in a `[[data_settings.files]]` entry.
const _FILE_ENTRY_KEYS = Set(["path", "format", "split", "variables",
                              "locations", "timerange", "source"])

function _validate_file_entry(entry::AbstractDict, i::Integer)
    for req in ("path", "format", "split", "variables")
        haskey(entry, req) ||
            error("load_data: file entry #$i is missing required key \"$req\".")
    end
    unknown = setdiff(Set(String(k) for k in keys(entry)), _FILE_ENTRY_KEYS)
    isempty(unknown) || error(
        "load_data: file entry #$i has unknown key(s): " *
        join(sort!(collect(unknown)), ", ") *
        ". Allowed: " * join(sort!(collect(_FILE_ENTRY_KEYS)), ", ") * ".")
    return nothing
end

"""
    load_data(toml_path::String) -> Dict{String, NamedTuple}

Convenience method: parse TOML from `toml_path` and call `load_data` with
`base_dir = dirname(toml_path)`.
"""
function load_data(toml_path::String)
    settings = TOML.parsefile(toml_path)
    load_data(settings["data_settings"]; base_dir=dirname(abspath(toml_path)))
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _load_entry!(flat::Dict{String,TimeSeries}, entry::Dict{String,Any}, base_dir::String)
    format    = entry["format"]
    raw_path  = entry["path"]
    path      = isabspath(raw_path) ? raw_path : joinpath(base_dir, raw_path)
    variables = _parse_variables(entry["variables"])

    if format == "noos"
        source     = entry["source"]
        collection = NoosTimeSeriesCollection(path)
        for (name, as) in variables
            ts = get_series_from_collection(collection, source, name)
            ts = _apply_filters(ts, entry)
            flat[as] = ts isa TimeSeries ? ts : TimeSeries(ts)
        end
    else
        for (name, as) in variables
            ts = _load_variable(format, path, name)
            ts = _apply_filters(ts, entry)
            flat[as] = ts isa TimeSeries ? ts : TimeSeries(ts)
        end
    end
end

function _apply_filters(ts, entry::Dict{String,Any})
    if haskey(entry, "timerange")
        tr = entry["timerange"]
        ts = select_timespan(ts, DateTime(tr[1]), DateTime(tr[2]))
    end
    if haskey(entry, "locations")
        ts = select_locations_by_names(ts, entry["locations"])
    end
    return ts
end

function _parse_variables(variables::Vector)
    map(variables) do v
        if v isa String
            (v, v)
        else
            (v["name"], get(v, "as", v["name"]))
        end
    end
end

function _load_variable(format::String, path::String, varname::String)
    if format == "netcdf"
        return TimeSeries(NetCDFTimeSeries(path, varname))
    elseif format == "jld2"
        return JLD2TimeSeries(path, varname=varname)
    else
        error("Unknown format: \"$format\". Supported: \"netcdf\", \"jld2\", \"noos\"")
    end
end

function _intersect_times(flat::Dict{String,TimeSeries}, split::AbstractString="")
    isempty(flat) && return flat
    t_start = maximum(get_times(ts)[1]   for ts in values(flat))
    t_end   = minimum(get_times(ts)[end] for ts in values(flat))
    if t_start > t_end
        loc = isempty(split) ? "" : " in split \"$split\""
        error("load_data: files$loc have no overlapping time window " *
              "(latest start $t_start is after earliest end $t_end).")
    end
    return Dict(k => select_timespan(ts, t_start, t_end) for (k, ts) in flat)
end
