module AIHydroPoints

# Packages used
using Dates
using MultiTimeSeries

export AbstractTimeSeries
export find_location_index
export TimeSeries

# Series I/O — now provided by MultiTimeSeries
export NetCDFTimeSeries, write_to_netcdf
export ZarrTimeSeries, has_aws_credentials
export JLD2TimeSeries, write_to_jld2
export NoosTimeSeriesCollection, write_single_noos_file, read_single_noos_file
export get_sources, get_source_quantity_keys, get_quantities, get_series_from_collection

include("tidal_comps.jl")
export primary_frequencies_as_doodson, get_doodson_eqvals, constituents, lunar2solar, robust_timedelta_sec

include("training.jl")
include("tides.jl")
export save_settings, load_settings, save_model, load_model, load_run, ModelSettings, prepare_train_data, predict, train_model, plot_losses, plot_series
export TideSettings, create_tide_model, TideModel

include("graph_network.jl")
export get_adjacency, GraphNetwork, plot_graph

include("attention.jl")
export Embedder, Deembedder, SinCosPosEmbedder, Transformer

include("surge.jl")
export SurgeSettings, create_surge_model, SurgeModel

include("interaction.jl")
export InteractionSettings, create_interaction_model

include("wind_stress.jl")
export uv_to_stress_xy

include("waves.jl")
export WaveSettings, create_wave_model, stats_skipnan, average_stats


# Methods in the interface for time series
# getters for the fields
export get_values, get_times, get_names, get_longitudes, get_latitudes, get_quantity, get_source
# selection methods
export select_locations_by_ids, select_location_by_id, select_locations_by_names, select_location_by_name, 
    select_timespan, select_times_by_ids
# tools
export merge_by_times, select_timerange_with_fill, merge_by_locations
# pretty printing/summary
export show

end # module series_ml

