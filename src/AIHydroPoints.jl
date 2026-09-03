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

include("toml_utils.jl")
export toml_write, toml_read

include("data_loading.jl")
export load_data

include("input_processing.jl")
export validate_and_augment_settings!

include("plot_utils.jl")
export save_loss_plot

include("maps.jl")
export plot_map

include("models/training_settings.jl")
export TrainingSettings, to_dict

include("models/abstract_model.jl")
export AbstractModel, get_settings, save_params, load_params!, train_model!, write_outputs

include("models/abstract_flux_model.jl")
export AbstractFluxModel, MyFluxModel
export preprocess, forward, postprocess!, get_flux_model, predict

include("models/AbstractSurgeModel.jl")
export AbstractSurgeModel, SurgeLagSource, nsamples, materialize_batch

include("models/LinearSurgeModel.jl")
export LinearSurgeModel

include("models/ConvSurgeModel.jl")
export ConvSurgeModel

include("models/AbstractSurgeInteractionModel.jl")
export AbstractSurgeInteractionModel

include("models/BiLinearSurgeInteractionModel.jl")
export BiLinearSurgeInteractionModel, BiLinearSurgeInteractionFlux, StationTideModulation

include("models/AbstractTideModel.jl")
export AbstractTideModel

include("models/DeepONetTideModel.jl")
export DeepONetTideModel, TideModel

include("models/ProductTideModel.jl")
export ProductTideModel, ProductTideFlux, ProductInputLayer, ProductGatingLayer

include("graph_network.jl")
export get_adjacency, GraphNetwork, plot_graph

include("attention.jl")
export Embedder, Deembedder, SinCosPosEmbedder, Transformer

include("models/AttentionSurgeModel.jl")
export AttentionSurgeModel, AttentionSurgeFlux


include("wind_stress.jl")
export uv_to_stress_xy

include("models/AbstractWaveModel.jl")
export AbstractWaveModel

include("models/ConvWaveModel.jl")
export ConvWaveModel, WaveInputLayer

include("models/DeepONetWaveModel.jl")
export DeepONetWaveModel, DeepONetWaveFlux

include("wave_stats.jl")
export stats_skipnan, average_stats

include("models/AbstractInteractionModel.jl")
export AbstractInteractionModel

include("models/ProductInteractionModel.jl")
export ProductInteractionModel, ProductInteractionFlux, InteractionInputLayer

include("models/ConvInteractionModel.jl")
export ConvInteractionModel, ConvInteractionFlux

include("model_registry.jl")
export MODEL_REGISTRY, get_model_type, validate_model_settings!, create_model

include("train.jl")
export train

include("predict.jl")
export predict

include("serve.jl")
export serve, load_served_model

include("leaderboard.jl")
export find_run_dirs, load_leaderboard, sort_leaderboard


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

