module AIHydroPoints

# Packages used
using Dates
using NetCDF

# Abstract data type for time series at multiple locations
include("abstract_series.jl")
export AbstractTimeSeries
export find_location_index, find_location_indices

# Time series tools
include("series.jl")
export TimeSeries

# NetCDF time series tools
include("series_netcdf.jl")
export NetCDFTimeSeries, write_to_netcdf

# Zarr time series tools
include("series_zarr.jl")
export ZarrTimeSeries

# Import and export for the JLD2 storage format
include("series_jld2.jl")
export JLD2TimeSeries, write_to_jld2

# NOOS time series tools
include("series_noos.jl")
export NoosTimeSeriesCollection, write_single_noos_file, read_single_noos_file
export get_sources, get_source_quantity_keys, get_quantities, get_series_from_collection

include("tidal_comps.jl")
export primary_frequencies_as_doodson, get_doodson_eqvals

include("netcdf_utils.jl")
export waterlevel_series_to_netcdf

include("training.jl")
include("tides.jl")
export save_settings, load_settings, save_model, load_model, load_run, ModelSettings, prepare_train_data, predict, train_model, plot_losses
export TideSettings, create_tide_model, prepare_train_data!, plot_series


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

