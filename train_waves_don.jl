# train_waves_don.jl
#
# Train a wave model using the DeepONet-on-stations architecture.
# Alternative to train_waves.jl: uses a dot-product station-modulation
# instead of WaveInputLayer's exponential channel-wise scaling.

cd(@__DIR__)
using Pkg
Pkg.activate(".")

using AIHydroPoints
using CUDA, cuDNN
using Dates
using Flux
using JLD2
using Plots
using ProgressMeter
using Statistics
using DataFrames
using CSV

# ──────────────────────────────────────────────
# GPU setup
# ──────────────────────────────────────────────
force_cpu = false
if CUDA.functional() && !force_cpu
    @info "CUDA functional"
    CUDA.allowscalar(false)
    device = gpu
else
    @info "Using CPU"
    device = cpu
end

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
# Model and optimizer settings
# ──────────────────────────────────────────────
nchannel     = (32, 32, 32, 16)
f_activation = swish
nlags        = 16
@assert nlags == 2^length(nchannel) "nlags must equal 2^length(nchannel)"
wind_scale   = 0.5f0
wave_scale   = 3.0f0

n_output_stations = length(output_locations)
n_input_stations  = length(input_locations)
n_input_vars      = 2  # wind_x, wind_y

nepochs              = 2    # FOR TESTING — increase for a real run (e.g. 50)
nbatch               = 256
learning_rate        = 0.001
learning_rate_decay  = 0.03
learning_rate_steps  = 10
regularization_weight = 1.0f-4
input_noise_std      = 0.30f0
validation_fractions = 0.25

runid      = "10to11_don2"
model_name = "wave_model_$(runid)_$(length(nchannel))lyr_$(nbatch)batch_$(nlags)lags_$(nepochs)epochs"

# ──────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────

"""
    times_to_timerange(times)

Convert a vector of DateTime to a regular DateTime range, estimating the step
from the median interval between consecutive times.
"""
function times_to_timerange(times)
    dt = median(Dates.value.(diff(times)))  # milliseconds
    return times[1]:Millisecond(round(Int, dt)):times[end]
end

"""
    prepare_wave_data(data_windspeed, data_winddirection, data_waveheight, nlags)

Prepare training arrays for the DeepONet wave model.

Returns `(x_station, x_input, y_wave)` where:
- `x_station`: one-hot station encoding `(n_stations, n_stations * n_training)`
- `x_input`: lagged wind stress blocks `(nlags, n_wind*2, n_stations * n_training)`
- `y_wave`: target wave heights `(1, n_stations * n_training)`
"""
function prepare_wave_data(data_windspeed, data_winddirection, data_waveheight, nlags)
    target_timerange = times_to_timerange(get_times(data_waveheight))
    wind_timerange   = times_to_timerange(get_times(data_winddirection))
    @assert step(target_timerange) == step(wind_timerange) "Time steps differ"
    @assert target_timerange[1]   == wind_timerange[1]     "Start times differ"
    @assert target_timerange[end] == wind_timerange[end]   "End times differ"

    itraining_times   = nlags:length(target_timerange)
    ntraining_times   = length(itraining_times)
    n_target          = length(get_names(data_waveheight))
    n_source          = length(get_names(data_windspeed))

    u10_values  = get_values(data_windspeed)
    udir_values = get_values(data_winddirection)
    swh_values  = get_values(data_waveheight)

    wind_x = Float32.(u10_values .* -sind.(udir_values) ./ wind_scale)
    wind_y = Float32.(u10_values .* -cosd.(udir_values) ./ wind_scale)
    for i in eachindex(wind_x)
        wind_x[i], wind_y[i] = uv_to_stress_xy(wind_x[i], wind_y[i])
    end
    swh_values = swh_values ./ wave_scale

    station_index = collect(1:n_target) * ones(Int, ntraining_times)'
    x_station = Flux.onehotbatch(station_index[:], 1:n_target)

    x_input = zeros(Float32, nlags, n_source * n_input_vars, n_target * ntraining_times)
    for itime in itraining_times
        x_block = Float32.(vcat(wind_x[:, itime-nlags+1:itime],
                                wind_y[:, itime-nlags+1:itime]))'
        for istation in 1:n_target
            isample = (itime - nlags) * n_target + istation
            x_input[:, :, isample] .= x_block
        end
    end

    y_wave = reshape(Float32.(swh_values[:, nlags:end]), 1, :)

    # Drop records with NaNs
    valid = [i for i in 1:size(y_wave, 2)
             if !any(isnan, x_input[:, :, i]) &&
                !any(isnan, x_station[:, i])  &&
                !isnan(y_wave[1, i])]
    return x_station[:, valid], x_input[:, :, valid], y_wave[:, valid]
end

@info "Preparing data"
x_station, x_input, y = prepare_wave_data(u10, udir, swh, nlags)

n_train = Int(floor((1.0 - validation_fractions) * size(x_input, 3)))
train_x_station_cpu = x_station[:, 1:n_train]
train_x_input_cpu   = x_input[:, :, 1:n_train]
train_y_cpu         = y[:, 1:n_train]
val_x_station_cpu   = x_station[:, n_train+1:end]
val_x_input_cpu     = x_input[:, :, n_train+1:end]
val_y_cpu           = y[:, n_train+1:end]
@show size(train_x_station_cpu), size(train_x_input_cpu), size(train_y_cpu)
@show size(val_x_station_cpu),   size(val_x_input_cpu),   size(val_y_cpu)

train_x_station = train_x_station_cpu |> device
train_x_input   = train_x_input_cpu   |> device
train_y         = train_y_cpu         |> device
val_x_station   = val_x_station_cpu   |> device
val_x_input     = val_x_input_cpu     |> device
val_y           = val_y_cpu           |> device

dataloader = Flux.DataLoader((train_x_station, train_x_input, train_y),
                              batchsize=nbatch, shuffle=true)

# ──────────────────────────────────────────────
# DeepONet-on-stations model
# ──────────────────────────────────────────────
# Station parameters provide a per-station weight vector; the branch net
# processes the wind time series. The output is the inner product of the two,
# collapsing the feature dimension to a scalar per sample.

struct DeepONet_on_stations{T1,T2}
    station_params::T1
    branch_net::T2
end

DeepONet_on_stations(n_out, nchannel, n_in_vars, n_in_stations, f_act) =
    DeepONet_on_stations(
        Dense(n_out => nchannel[end], relu; bias=false),
        Chain(
            Conv((2,), (n_in_stations * n_in_vars) => nchannel[1], f_act, stride=(2,)),
            Conv((2,), nchannel[1] => nchannel[2], f_act, stride=(2,)),
            Conv((2,), nchannel[2] => nchannel[3], f_act, stride=(2,)),
            Conv((2,), nchannel[3] => nchannel[4], identity, stride=(2,)),
            Flux.flatten,
        ),
    )

function (l::DeepONet_on_stations)(x)
    x_station, x_input = x
    x1 = l.branch_net(x_input)
    s1 = l.station_params(x_station)
    s1 = reshape(s1, size(x1))
    return sum(s1 .* x1, dims=1)
end

Flux.@layer DeepONet_on_stations

model = DeepONet_on_stations(n_output_stations, nchannel, n_input_vars,
                              n_input_stations, f_activation) |> device

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
function compute_loss(model, x_station, x_input, y)
    return Flux.mse(model((x_station, x_input)), y)
end

train_losses = Float32[]
val_losses   = Float32[]
acc_losses   = Float32[]
opt_state    = Flux.setup(OptimiserChain(WeightDecay(regularization_weight),
                                         Adam(learning_rate)), model)

@info "Start training"
@showprogress for epoch in 1:nepochs
    acc_loss = 0.0f0
    for (xs, xi, yb) in dataloader
        xi_noisy = input_noise_std > 0.0f0 ?
            xi .+ input_noise_std .* (xi isa CuArray ?
                CUDA.randn(Float32, size(xi)...) :
                randn(Float32, size(xi))) : xi
        dloss, grads = Flux.withgradient(model) do m
            Flux.mse(m((xs, xi_noisy)), yb)
        end
        Flux.update!(opt_state, model, grads[1])
        acc_loss += dloss
    end
    push!(acc_losses, acc_loss)

    model_cpu = model |> cpu
    push!(train_losses, compute_loss(model_cpu, train_x_station_cpu, train_x_input_cpu, train_y_cpu))
    push!(val_losses,   compute_loss(model_cpu, val_x_station_cpu,   val_x_input_cpu,   val_y_cpu))
    @info "Epoch $epoch  train=$(train_losses[end])  val=$(val_losses[end])"

    if epoch % learning_rate_steps == 0
        Flux.adjust!(opt_state, learning_rate * learning_rate_decay^(epoch / nepochs))
    end
end

# ──────────────────────────────────────────────
# Save model and losses plot
# ──────────────────────────────────────────────
if isdir(model_name)
    rm(model_name, recursive=true, force=true)
end
mkdir(model_name)

model_cpu = model |> cpu
save(joinpath(model_name, "$(model_name).jld2"), "model", model_cpu)

plot(train_losses, label="training loss", xlabel="Epoch", ylabel="Loss")
plot!(val_losses, label="validation loss")
savefig(joinpath(model_name, "losses_train_val.png"))

# ──────────────────────────────────────────────
# Predict on full dataset and save
# ──────────────────────────────────────────────

"""
    predict_don(model, wind_speed, wind_dir, wave_ref)

Run the DeepONet model over the full time range and return predictions as a TimeSeries.
"""
function predict_don(model, wind_speed, wind_dir, wave_ref)
    output_names      = get_names(wave_ref)
    output_longitudes = copy(get_longitudes(wave_ref))
    output_latitudes  = copy(get_latitudes(wave_ref))

    times          = get_times(wind_speed)
    n_times        = length(times)
    itimes         = nlags:n_times
    ntraining      = length(itimes)
    n_in_stations  = length(get_names(wind_speed))

    u10_vals  = get_values(wind_speed)
    udir_vals = get_values(wind_dir)
    wind_x    = Float32.(u10_vals .* -sind.(udir_vals) ./ wind_scale)
    wind_y    = Float32.(u10_vals .* -cosd.(udir_vals) ./ wind_scale)
    for i in eachindex(wind_x)
        wind_x[i], wind_y[i] = uv_to_stress_xy(wind_x[i], wind_y[i])
    end

    x_input = zeros(Float32, nlags, n_in_stations * n_input_vars, ntraining)
    for itime in itimes
        x_input[:, :, itime-nlags+1] .= Float32.(
            vcat(wind_x[:, itime-nlags+1:itime],
                 wind_y[:, itime-nlags+1:itime]))'
    end

    n_out  = length(output_names)
    y_hat  = zeros(Float32, n_out, n_times)
    y_hat[:, 1:nlags-1] .= NaN
    for istation in 1:n_out
        x_station = Flux.onehotbatch(fill(istation, ntraining), 1:n_out)
        y_hat[istation, nlags:end] .= wave_scale .* model((x_station, x_input))[1, :]
    end

    return TimeSeries(y_hat, times, output_names, output_longitudes, output_latitudes,
                      "wave_height", "DeepONet wave model")
end

swh_predicted = predict_don(model_cpu, u10, udir, swh)
write_to_jld2(swh_predicted,   joinpath(model_name, "predicted_wave_heights.jld2"))
write_to_netcdf(swh_predicted, joinpath(model_name, "predicted_wave_heights.nc"))

# ──────────────────────────────────────────────
# Per-timespan statistics and plots
# ──────────────────────────────────────────────
function plot_series_don(y_true::TimeSeries, y_pred::TimeSeries, station_name)
    quantity = replace(get_quantity(y_true), "_" => " ")
    y_true_s = select_location_by_name(y_true, station_name)
    y_pred_s = select_location_by_name(y_pred, station_name)
    times    = get_times(y_true_s)
    p = plot(times, vec(get_values(y_pred_s)), label="predicted", color=:blue,
             xlabel="Time", ylabel=quantity, title="Station: $station_name")
    plot!(p, times, vec(get_values(y_true_s)), label="target", color=:black)
    return p
end

function plot_scatter_don(y_true::TimeSeries, y_pred::TimeSeries, station_name)
    quantity = replace(get_quantity(y_true), "_" => " ")
    y_true_s = select_location_by_name(y_true, station_name)
    y_pred_s = select_location_by_name(y_pred, station_name)
    return scatter(vec(get_values(y_true_s)), vec(get_values(y_pred_s)),
                   label=false, color=:black, ms=1,
                   xlabel="target $quantity", ylabel="predicted $quantity",
                   title="Scatter: $station_name")
end

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
    CSV.write(joinpath(model_name, "$(timespan_name)_statistics_wave_height.csv"), stats)

    for name in get_names(swh_ts)
        savefig(plot_series_don(swh_ts, swh_pred, name),
                joinpath(model_name, "$(timespan_name)_wave_height_$(name).png"))
        savefig(plot_scatter_don(swh_ts, swh_pred, name),
                joinpath(model_name, "$(timespan_name)_scatter_$(name).png"))
    end
end

@show avg_stats_df
CSV.write(joinpath(model_name, "average_statistics_wave_height.csv"), avg_stats_df)
