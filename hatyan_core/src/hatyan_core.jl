module hatyan_core

# ── dependencies ─────────────────────────────────────────────────────────────
using Dates
using Printf
using MultiTimeSeries

# ── abstract types ────────────────────────────────────────────────────────────
abstract type AbstractTidalConstituents end

# ── source files ──────────────────────────────────────────────────────────────
include("constituents.jl")
include("constituents_donar.jl")
include("constituent_list.jl")
include("settings.jl")
include("doodson.jl")
include("schureman.jl")
include("foreman.jl")
include("analysis.jl")
include("prediction.jl")

# ── exports ───────────────────────────────────────────────────────────────────

# re-export everything from MultiTimeSeries
export AbstractTimeSeries
export TimeSeries
export NoosTimeSeriesCollection
export get_values, get_times, get_names, get_longitudes, get_latitudes
export get_quantity, get_source
export find_location_index
export select_location_by_id, select_locations_by_ids
export select_location_by_name, select_locations_by_names
export select_timespan, select_timerange_with_fill, select_times_by_ids
export merge_by_times, merge_by_locations
export read_single_noos_file, read_muliple_noos_files, write_single_noos_file
export get_source_quantity_keys, get_sources, get_quantities, get_series_from_collection
export read_donar_timeseries

# tidal constituents
export AbstractTidalConstituents
export TidalConstituents
export get_amplitudes, get_phases, get_constituent_names
export select_constituents_by_names

# constituent DONAR I/O
export read_donar_constituents

# Constituent lists
export constituent_list

# Settings
export HatyanSettings

# Doodson astronomical arguments
export DOOD_T, DOOD_S, DOOD_H, DOOD_P, DOOD_N, DOOD_P1
export robust_timedelta_sec, get_doodson_eqvals

# Schureman constituent tables
export SchuremanTable, get_schureman_table
export get_schureman_freqs, get_schureman_v0
export get_schureman_constants, get_schureman_u, get_schureman_f
export correct_fwith_xfac

# Foreman constituent tables
export ForemanTable, get_foreman_table
export get_foreman_freqs, get_foreman_v0, get_foreman_uf

# Generic dispatch wrappers (method="schureman" or method="foreman")
export get_freqv0_generic, get_uf_generic

# Analysis
export analysis

# Prediction
export prediction

end # module hatyan_core
