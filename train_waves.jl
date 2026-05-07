# new_train_waves.jl
#
# Train a wave model on the waves_2021 test dataset (NOOS format).
#
# Select which model to train by changing model_type below:
#   "ConvWaveModel"       — WaveInputLayer + strided Conv (exp channel modulation)
#   "DeepONetWaveModel"   — strided Conv branch + dot-product station merge
#
# The model predicts significant wave height from wind speed and direction.

cd(@__DIR__)

using Pkg
Pkg.activate(".")

model_type = "ConvWaveModel"   # "ConvWaveModel" | "DeepONetWaveModel"
model_type = "DeepONetWaveModel"  

ENV["GKSwstype"] = "nul"   # headless GR backend
using AIHydroPoints
using Dates

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "test_data", "waves_2021")
series_collection = NoosTimeSeriesCollection(data_dir)

u10  = get_series_from_collection(series_collection, "knmi_harmonie40_wind", "wind_speed")
udir = get_series_from_collection(series_collection, "knmi_harmonie40_wind", "wind_direction")
swh  = get_series_from_collection(series_collection, "swan_dcsm_harmonie",   "wave_height")

time_selection = DateTime(2021, 1, 1):Hour(1):DateTime(2021, 12, 31, 23)
u10  = select_timerange_with_fill(u10,  time_selection, fill_value=0.0f0)
udir = select_timerange_with_fill(udir, time_selection, fill_value=0.0f0)
swh  = select_timerange_with_fill(swh,  time_selection, fill_value=0.0f0)

# ──────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────
t_split = DateTime(2021, 10, 1)

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

nstations = length(get_names(swh))
nwind     = length(get_names(u10))

# ──────────────────────────────────────────────
# Model / training settings
# ──────────────────────────────────────────────
name     = model_type
save_dir = joinpath("models", name)

rm(save_dir, recursive=true, force=true)
mkpath(save_dir)

model_settings = Dict{String, Any}(
    "model_name" => name,
    "model_dir"  => save_dir,
    "nstations"  => nstations,
    "nwind"      => nwind,
    "nlags"      => 16,
    "wind_scale" => 0.5,
    "wave_scale" => 3.0,
)

if model_type == "ConvWaveModel"
    model_settings["n_input_channels"] = 64
    model_settings["model_pars"] = Dict{String, Any}(
        "nchannel"   => [64, 64, 64, 1],
        "activation" => "swish",
    )
elseif model_type == "DeepONetWaveModel"
    model_settings["model_pars"] = Dict{String, Any}(
        "nchannel"   => [32, 32, 32, 16],
        "activation" => "swish",
    )
end

train_settings = TrainingSettings(
    nepochs          = 50,      # 2 FOR TESTING — increase for a real run (e.g. 50)
    nbatches         = 256,
    learning_rate    = 1.0e-3,
    lr_decay_factor  = 0.1,
    lr_decay_rate    = 400,
    patience         = 5,
    validation_split = 0.2,
)

# ──────────────────────────────────────────────
# Create and train model
# ──────────────────────────────────────────────
if model_type == "ConvWaveModel"
    model = ConvWaveModel(model_settings)
elseif model_type == "DeepONetWaveModel"
    model = DeepONetWaveModel(model_settings)
else
    error("Unknown model_type: $model_type")
end

train_losses, val_losses = train_model!(model, train_settings, train_dict, train_dict)

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
save_params(model, joinpath(save_dir, "params.jld2"); overwrite=true)
toml_write(joinpath(save_dir, "settings.toml"), get_settings(model); overwrite=true)
save_loss_plot(joinpath(save_dir, "losses.png"), train_losses, val_losses; overwrite=true)

# ──────────────────────────────────────────────
# Evaluation plots on test set
# ──────────────────────────────────────────────
plot_series(model, test_dict, test_dict, "test")
