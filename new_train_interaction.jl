# new_train_interaction.jl
#
# Train a tide-surge interaction model using the Schureman 2011 tide and surge
# test datasets.
#
# Waterlevel is synthesised as tide + surge so the model learns to reconstruct
# the linear combination.  In a real application, replace this with observed
# waterlevel.

cd(@__DIR__)

using Pkg
Pkg.activate(".")

ENV["GKSwstype"] = "nul"   # headless GR backend

using AIHydroPoints
using Dates

# ──────────────────────────────────────────────
# File paths
# ──────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "test_data")

#NOTE: this surge was computed as waterlevel - tide and thus contains much of the interaction already. It's just a synthetic test for the code, not a real test of the model's ability to learn the interaction from separate tide and surge inputs.
tide_train_file  = joinpath(data_dir, "tides_schureman_2011.nc")
surge_train_file = joinpath(data_dir, "surge_schureman_2011.nc") 
tide_val_file    = joinpath(data_dir, "tides_schureman_2012.nc")
surge_val_file   = joinpath(data_dir, "surge_schureman_2012.nc")

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
ts_tide_train  = TimeSeries(NetCDFTimeSeries(tide_train_file,  "waterlevel"))
ts_surge_train = TimeSeries(NetCDFTimeSeries(surge_train_file, "surge"))
ts_tide_val    = TimeSeries(NetCDFTimeSeries(tide_val_file,    "waterlevel"))
ts_surge_val   = TimeSeries(NetCDFTimeSeries(surge_val_file,   "surge"))

# Synthesise waterlevel = tide + surge (proxy for observed waterlevel)
wl_train_vals = get_values(ts_tide_train) .+ get_values(ts_surge_train)
wl_val_vals   = get_values(ts_tide_val)   .+ get_values(ts_surge_val)

ts_wl_train = TimeSeries(
    Float32.(wl_train_vals),
    get_times(ts_tide_train),
    get_names(ts_tide_train),
    Float64.(get_longitudes(ts_tide_train)),
    Float64.(get_latitudes(ts_tide_train)),
    "waterlevel",
    "synthetic",
)
ts_wl_val = TimeSeries(
    Float32.(wl_val_vals),
    get_times(ts_tide_val),
    get_names(ts_tide_val),
    Float64.(get_longitudes(ts_tide_val)),
    Float64.(get_latitudes(ts_tide_val)),
    "waterlevel",
    "synthetic",
)

train_input  = Dict{String, TimeSeries}("tide" => ts_tide_train, "surge" => ts_surge_train)
train_target = Dict{String, TimeSeries}("waterlevel" => ts_wl_train)
val_input    = Dict{String, TimeSeries}("tide" => ts_tide_val,   "surge" => ts_surge_val)
val_target   = Dict{String, TimeSeries}("waterlevel" => ts_wl_val)

@info "Loaded training data" nstations=size(get_values(ts_tide_train), 1) ntimes=size(get_values(ts_tide_train), 2)

# ──────────────────────────────────────────────
# Model / training settings
# ──────────────────────────────────────────────
save_dir = joinpath("models", "ConvInteractionModel")

rm(save_dir, recursive=true, force=true)
mkpath(save_dir)

model_settings = Dict{String, Any}(
    "model_name" => "ConvInteractionModel",
    "model_dir"  => save_dir,
    "nstations"  => size(get_values(ts_tide_train), 1),
    "nlags"      => 16,
    "model_pars" => Dict{String, Any}("channels" => [64, 32, 16, 1]),
)

train_settings = TrainingSettings(
    nepochs          = 2,    # increase for a real run (e.g. 200)
    nbatches         = 64,
    learning_rate    = 1e-3,
    validation_split = 0.2,
)

# ──────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────
model = ConvInteractionModel(model_settings)
@info "Training $(model_settings["model_name"])"

train_losses, val_losses = train_model!(model, train_settings, train_input, train_target)

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
params_file = joinpath(save_dir, "model_params.jld2")
save_params(model, params_file)
@info "Saved parameters" file=params_file

toml_write(joinpath(save_dir, "settings.toml"),get_settings(model); overwrite=true)

save_loss_plot(joinpath(save_dir, "losses.png"),train_losses, val_losses; overwrite=true)

# ──────────────────────────────────────────────
# Evaluate and plot
# ──────────────────────────────────────────────
plot_series(model, train_input, train_target, "train"; save_dir)
plot_series(model, val_input,   val_target,   "val";   save_dir)

@info "Done"
