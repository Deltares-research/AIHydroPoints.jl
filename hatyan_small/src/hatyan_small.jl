module hatyan_small

# ── dependencies ─────────────────────────────────────────────────────────────
using Dates
using Printf

# ── abstract types (must precede includes) ───────────────────────────────────
abstract type AbstractTimeSeries end
abstract type AbstractTidalConstituents end

# ── source files ─────────────────────────────────────────────────────────────
include("series.jl")
include("series_noos.jl")
include("series_donar.jl")
include("constituents.jl")
include("constituents_donar.jl")

# ── exports ───────────────────────────────────────────────────────────────────

# types
export AbstractTimeSeries
export TimeSeries
export NoosTimeSeriesCollection

# getters
export get_values, get_times, get_names, get_longitudes, get_latitudes
export get_quantity, get_source

# selection
export select_location_by_id, select_locations_by_ids
export select_location_by_name, select_locations_by_names
export select_timespan, select_timerange_with_fill, select_times_by_ids

# merging
export merge_by_times, merge_by_locations

# NOOS I/O
export read_single_noos_file, read_muliple_noos_files, write_single_noos_file
export get_source_quantity_keys, get_sources, get_quantities, get_series_from_collection

# DONAR I/O
export read_donar_timeseries

# tidal constituents
export AbstractTidalConstituents
export TidalConstituents
export get_amplitudes, get_phases, get_constituent_names
export select_constituents_by_names

# constituent DONAR I/O
export read_donar_constituents

end # module hatyan_small
