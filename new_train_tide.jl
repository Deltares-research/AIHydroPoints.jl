# Train a tide model on the Schureman 2011 tide dataset.
#
# Select which model to train by changing model_type below:
#   "DeepONetTideModel" — branch/trunk DeepONet architecture
#   "ProductTideModel"  — multiplicative station × Doodson product with gating layers
#
# The model is astronomically driven — inputs are Doodson numbers computed from
# time and station coordinates.  Both input and target are the same waterlevel
# TimeSeries.

cd(@__DIR__)

using Pkg
Pkg.activate(".")

model_type = "DeepONetTideModel"   # "DeepONetTideModel" | "ProductTideModel"

ENV["GKSwstype"] = "nul"   # headless GR backend (no display needed)
using AIHydroPoints

# ──────────────────────────────────────────────
# File paths
# ──────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "test_data")

filenames = Dict(
    "training" => joinpath(data_dir, "tides_schureman_2011.nc"),
    "testing"  => joinpath(data_dir, "tides_schureman_2012.nc"),
)

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
    "freqs"      => ["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H"],
)

if model_type == "DeepONetTideModel"
    model_settings["model_pars"] = Dict{String, Any}(
        "nlayers_branch" => 2,
        "nhidden_branch" => 64,
        "nlayers_trunk"  => 2,
        "nhidden_trunk"  => 32,
        "nlayers_down"   => 1,
    )
elseif model_type == "ProductTideModel"
    model_settings["model_pars"] = Dict{String, Any}(
        "nfeats"  => 64,
        "nlayers" => 4,
    )
end

train_settings = TrainingSettings(
    nepochs          = 500,
    nbatches         = 64,
    learning_rate    = 5.0e-3,
    lr_decay_factor  = 0.1,
    lr_decay_rate    = 400,
    patience         = 5,
    validation_split = 0.2,
)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
ts_train = TimeSeries(NetCDFTimeSeries(filenames["training"], "waterlevel"))
ts_test  = TimeSeries(NetCDFTimeSeries(filenames["testing"],  "waterlevel"))

train_data = Dict{String, TimeSeries}("waterlevel" => ts_train)
test_data  = Dict{String, TimeSeries}("waterlevel" => ts_test)

# ──────────────────────────────────────────────
# Create and train model
# ──────────────────────────────────────────────
if model_type == "DeepONetTideModel"
    model = DeepONetTideModel(model_settings)
elseif model_type == "ProductTideModel"
    model = ProductTideModel(model_settings)
else
    error("Unknown model_type: $model_type")
end

train_losses, val_losses = train_model!(model, train_settings, train_data, train_data)

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
save_params(model, joinpath(save_dir, "params.jld2"); overwrite=true)
toml_write(get_settings(model),joinpath(save_dir, "settings.toml"); overwrite=true)

# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
save_loss_plot(joinpath(save_dir, "losses.png"), train_losses, val_losses; overwrite=true)

# ──────────────────────────────────────────────
# Run inference on test set
# ──────────────────────────────────────────────
test_output = predict(model, test_data)

# ──────────────────────────────────────────────
# Evaluation plots
# ──────────────────────────────────────────────
plot_series(model, test_data, test_data, "test")