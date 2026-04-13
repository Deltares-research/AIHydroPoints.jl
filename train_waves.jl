# train_waves.jl
#
# Train a model to predict wave heights from wind stress.

cd(@__DIR__)
using Pkg
Pkg.activate(".")

using AIHydroPoints
using CUDA
using Dates
using Plots
using Statistics
using CSV

# ──────────────────────────────────────────────
# Load and align input data
# ──────────────────────────────────────────────
data_folder = joinpath(@__DIR__, "data", "waves_2021_2024_10to11")
series_collection = NoosTimeSeriesCollection(data_folder)
@show get_sources(series_collection)
@show get_quantities(series_collection)

u10  = get_series_from_collection(series_collection, "knmi_harmonie40_wind", "wind_speed")
udir = get_series_from_collection(series_collection, "knmi_harmonie40_wind", "wind_direction")
swh  = get_series_from_collection(series_collection, "swan_dcsm_harmonie",   "wave_height")

output_locations = get_names(swh)
input_locations  = get_names(u10)
@show output_locations
@show input_locations

u10  = select_locations_by_names(u10,  input_locations)
udir = select_locations_by_names(udir, input_locations)
swh  = select_locations_by_names(swh,  output_locations)

time_selection = DateTime(2021,1,1):Hour(1):DateTime(2024,11,1,0)
u10  = select_timerange_with_fill(u10,  time_selection, fill_value=0.0f0)
udir = select_timerange_with_fill(udir, time_selection, fill_value=0.0f0)
swh  = select_timerange_with_fill(swh,  time_selection, fill_value=0.0f0)

# ──────────────────────────────────────────────
# Train / validation split
# ──────────────────────────────────────────────
validation_fraction = 0.25
n_train = Int(floor((1.0 - validation_fraction) * length(time_selection)))
t_split = time_selection[n_train]

train_dict = Dict(
    "wind_speed"     => select_timespan(u10,  time_selection[1], t_split),
    "wind_direction" => select_timespan(udir, time_selection[1], t_split),
    "wave_height"    => select_timespan(swh,  time_selection[1], t_split),
)
test_dict = Dict(
    "wind_speed"     => select_timespan(u10,  t_split, time_selection[end]),
    "wind_direction" => select_timespan(udir, t_split, time_selection[end]),
    "wave_height"    => select_timespan(swh,  t_split, time_selection[end]),
)

# ──────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────
runid    = "10to11_explyr3"
settings = WaveSettings(
    model_name       = "wave_model_$(runid)_3stations",
    model_dir        = "wave_model_$(runid)_3stations",
    use_gpu          = CUDA.functional(),
    nstations        = length(output_locations),
    nwind            = length(input_locations),
    nlags            = 16,
    n_input_channels = 64,
    wind_scale       = 0.5,
    wave_scale       = 3.0,
    model_pars       = Dict("nchannel" => [64, 64, 64, 1], "activation" => "swish"),
)

train_settings = TrainingSettings(
    nepochs         = 2,    # FOR TESTING — change to 100 for a real run
    # nepochs       = 100,
    nbatches        = 256,
    learning_rate   = 0.001,
    weight_reg      = 1.0e-4,
    input_noise_std = 0.30,
)

if isdir(settings.model_dir)
    rm(settings.model_dir, recursive=true, force=true)
end
mkdir(settings.model_dir)

# ──────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────
model = create_wave_model(settings)
model, acc_losses, train_losses, test_losses =
    train_model(model, settings, train_settings, train_dict, test_dict)
save_model(model, settings)
save_settings(settings, train_settings)
plot_losses(train_losses, test_losses, settings, train_settings)

# ──────────────────────────────────────────────
# Predict on full dataset and save
# ──────────────────────────────────────────────
full_dict = Dict("wind_speed" => u10, "wind_direction" => udir, "wave_height" => swh)
swh_predicted = predict(model, settings, full_dict)

write_to_jld2(swh_predicted,   joinpath(settings.model_dir, "predicted_wave_heights.jld2"))
write_to_netcdf(swh_predicted, joinpath(settings.model_dir, "predicted_wave_heights.nc"))

# ──────────────────────────────────────────────
# Per-timespan statistics and plots
# ──────────────────────────────────────────────
timespans = Dict(
    "training" => (DateTime(2021,1,1),   DateTime(2023,12,31,23)),
    "test"     => (DateTime(2024,1,1),   DateTime(2024,11,1)),
    "202401"   => (DateTime(2024,1,1),   DateTime(2024,2,1,0)),
)

avg_stats_df = nothing
for (timespan_name, (tstart, tend)) in timespans
    @info "Statistics for timespan: $timespan_name ($tstart → $tend)"

    swh_ts   = select_timespan(swh,           tstart, tend)
    swh_pred = select_timespan(swh_predicted, tstart, tend)
    stats    = stats_skipnan(swh_ts, swh_pred)
    @show stats

    global avg_stats_df = average_stats(avg_stats_df, stats, timespan_name)
    CSV.write(joinpath(settings.model_dir, "$(timespan_name)_statistics_wave_height.csv"), stats)

    span_dict = Dict(
        "wind_speed"     => select_timespan(u10,  tstart, tend),
        "wind_direction" => select_timespan(udir, tstart, tend),
        "wave_height"    => swh_ts,
    )
    plot_series(model, settings, span_dict, "$(timespan_name)_wave_height")
end

@show avg_stats_df
CSV.write(joinpath(settings.model_dir, "average_statistics_wave_height.csv"), avg_stats_df)
