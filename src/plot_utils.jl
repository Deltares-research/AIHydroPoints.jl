using Plots
using FFTW
using Statistics
using Printf: @sprintf

# ──────────────────────────────────────────────────────────────────────────────
# FFT plotting helpers (used by _plot_station_series and by tides.jl)
# ──────────────────────────────────────────────────────────────────────────────

function plot_fft(signal, times, label)
    n  = length(signal)
    dt = (times[2] - times[1]).value / 3.6e6
    fft_out = fftshift(FFTW.fft(signal)) * 2 / n
    freqs   = fftshift(fftfreq(n, 1/dt))
    plot(freqs, abs.(fft_out); xlabel="Frequency (1/Hrs)", ylabel="Amplitude",
         xlims=(0, 0.5), label)
end

function plot_fft!(fig, signal, times, label)
    n  = length(signal)
    dt = (times[2] - times[1]).value / 3.6e6
    fft_out = fftshift(FFTW.fft(signal)) * 2 / n
    freqs   = fftshift(fftfreq(n, 1/dt))
    plot!(fig, freqs, abs.(fft_out); xlabel="Frequency (1/Hrs)", ylabel="Amplitude",
          xlims=(0, 0.5), label)
end

# ──────────────────────────────────────────────────────────────────────────────
# Shared per-station prediction-vs-truth plotting skeleton
# ──────────────────────────────────────────────────────────────────────────────

"""
    _plot_station_series(output, target, save_dir;
                         timerange=nothing, station_names=nothing)

Internal helper used by `write_outputs`.  Compares `output` from `predict` to
`target` ground truth per station and saves one PNG per station to `save_dir`
(which must already exist).

`output` and `target` must share the same primary key (e.g. `"surge"` or
`"waterlevel"`).  Target times are aligned to the output times automatically,
which handles lag trimming in surge models.
"""
function _plot_station_series(output::Dict{String, TimeSeries},
                               target::Dict{String, TimeSeries},
                               save_dir::String;
                               timerange     = nothing,
                               station_names = nothing)

    out_key = first(keys(output))
    ts_pred = output[out_key]
    ts_true = target[out_key]

    # Align ground truth to prediction times (handles lag trimming in surge models)
    t_start = get_times(ts_pred)[1]
    t_end   = get_times(ts_pred)[end]
    ts_true = select_timespan(ts_true, t_start, t_end)

    if !isnothing(station_names)
        ts_pred = select_locations_by_names(ts_pred, station_names)
        ts_true = select_locations_by_names(ts_true, station_names)
    end
    if !isnothing(timerange)
        ts_pred = select_timespan(ts_pred, timerange[1], timerange[2])
        ts_true = select_timespan(ts_true, timerange[1], timerange[2])
    end

    pred   = get_values(ts_pred)           # (nstations, ntimes)
    truth  = get_values(ts_true)
    times  = get_times(ts_pred)
    names  = get_names(ts_pred)
    errors = truth .- pred
    rmses  = sqrt.(mean(abs2, errors; dims=2))[:, 1]

    qty = get_quantity(ts_true)

    for (i, station) in enumerate(names)
        rmse_str = @sprintf("%.4f", rmses[i])
        p1 = plot(ts_true; location_index=i, label="Observations",
                  title="$station  RMSE = $rmse_str")
        plot!(p1, times, pred[i, :]; label="Predicted")
        p2 = plot(times, errors[i, :]; label="Residual", xlabel="Time", ylabel=qty)

        plot(p1, p2; layout=(2, 1), size=(800, 600))
        savefig(joinpath(save_dir, "$(station).png"))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Scatter plot helper
# ──────────────────────────────────────────────────────────────────────────────

"""
    _plot_station_scatter(output, target, save_dir;
                          timerange=nothing, station_names=nothing)

Internal helper used by `write_outputs`.  Produces a predicted-vs-observed
scatter plot per station and saves one PNG per station to `save_dir`
(which must already exist).
"""
function _plot_station_scatter(output::Dict{String, TimeSeries},
                                target::Dict{String, TimeSeries},
                                save_dir::String;
                                timerange     = nothing,
                                station_names = nothing)

    out_key = first(keys(output))
    ts_pred = output[out_key]
    ts_true = target[out_key]

    t_start = get_times(ts_pred)[1]
    t_end   = get_times(ts_pred)[end]
    ts_true = select_timespan(ts_true, t_start, t_end)

    if !isnothing(station_names)
        ts_pred = select_locations_by_names(ts_pred, station_names)
        ts_true = select_locations_by_names(ts_true, station_names)
    end
    if !isnothing(timerange)
        ts_pred = select_timespan(ts_pred, timerange[1], timerange[2])
        ts_true = select_timespan(ts_true, timerange[1], timerange[2])
    end

    names = get_names(ts_pred)
    for (i, station) in enumerate(names)
        p = scatter(ts_true, ts_pred; location_index=i)
        savefig(p, joinpath(save_dir, "$(station).png"))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Loss plot
# ──────────────────────────────────────────────────────────────────────────────

"""
    save_loss_plot(path::String, train_losses::Vector, val_losses::Vector=[];
                   overwrite::Bool=false)

Plot `train_losses` (and optionally `val_losses`) against epoch number and save
the figure as a PNG to `path`.

Throws an error if the parent directory does not exist, or if the file already
exists and `overwrite` is `false`.
"""
function save_loss_plot(path::String, train_losses::Vector, val_losses::Vector=[];
                        overwrite::Bool=false)
    isdir(dirname(path)) || error("directory does not exist: $(dirname(path))")
    !overwrite && isfile(path) && error("file already exists (use overwrite=true): $path")

    epochs = 1:length(train_losses)
    p = plot(epochs, train_losses; label="train RMSE", xlabel="epoch", ylabel="RMSE",
             title="Training losses")
    isempty(val_losses) || plot!(p, epochs, val_losses; label="val RMSE")
    savefig(p, path)
end
