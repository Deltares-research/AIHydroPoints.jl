using Plots
using Statistics
using Printf: @sprintf
using CSV
using hatyan_core: fft_series, TimeSeries, get_values, get_times, get_names,
                   get_longitudes, get_latitudes, get_quantity, get_source,
                   get_frequencies, get_amplitudes,
                   analysis, get_constituent_names, constituent_list

# ──────────────────────────────────────────────────────────────────────────────
# FFT plot helper
# ──────────────────────────────────────────────────────────────────────────────

"""
    _plot_station_fft(output, target, save_dir;
                      timerange=nothing, station_names=nothing)

Internal helper used by `write_outputs`.  Produces a 2-panel FFT plot per
station (observed + predicted spectra on top, residual spectrum below) and
saves one PNG per station to `save_dir` (which must already exist).
"""
function _plot_station_fft(output::Dict{String, TimeSeries},
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
    ts_true = _check_and_align_locations(ts_true, get_names(ts_pred), "target[\"$out_key\"]")

    if !isnothing(station_names)
        ts_pred = select_locations_by_names(ts_pred, station_names)
        ts_true = select_locations_by_names(ts_true, station_names)
    end
    if !isnothing(timerange)
        ts_pred = select_timespan(ts_pred, timerange[1], timerange[2])
        ts_true = select_timespan(ts_true, timerange[1], timerange[2])
    end

    residual_vals = get_values(ts_true) .- get_values(ts_pred)
    ts_residual   = TimeSeries(residual_vals, get_times(ts_pred),
                               get_names(ts_pred), get_longitudes(ts_pred),
                               get_latitudes(ts_pred), get_quantity(ts_pred),
                               get_source(ts_pred) * " | residual")

    fs_true     = fft_series(ts_true)
    fs_pred     = fft_series(ts_pred)
    fs_residual = fft_series(ts_residual)
    names       = get_names(ts_pred)
    freqs_cpd   = get_frequencies(fs_true) .* 86400.0   # Hz → cycles/day

    for (i, station) in enumerate(names)
        # observations + predicted overlaid on one panel
        p1 = plot(fs_true; location_index=i, label="Observations")
        plot!(p1, freqs_cpd, get_amplitudes(fs_pred)[i, :]; label="Predicted")
        p2 = plot(fs_residual; location_index=i, label="Residual")
        plot(p1, p2; layout=(2, 1), size=(800, 600))
        savefig(joinpath(save_dir, "$(station).png"))
    end
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
    ts_true = _check_and_align_locations(ts_true, get_names(ts_pred), "target[\"$out_key\"]")

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
                          timerange=nothing, station_names=nothing,
                          add_fit=false, add_qq=false)

Internal helper used by `write_outputs`.  Produces a predicted-vs-observed
scatter plot per station and saves one PNG per station to `save_dir`
(which must already exist).  The scatter renders at `density=:auto` (transparent
points for small series, a density heatmap for large ones).  Optional overlays:
`add_fit` draws a least-squares fit line and an r/slope/offset/bias stats box;
`add_qq` overlays a quantile-quantile curve with labelled percentile dots.
"""
function _plot_station_scatter(output::Dict{String, TimeSeries},
                                target::Dict{String, TimeSeries},
                                save_dir::String;
                                timerange     = nothing,
                                station_names = nothing,
                                add_fit::Bool = false,
                                add_qq::Bool  = false)

    out_key = first(keys(output))
    ts_pred = output[out_key]
    ts_true = target[out_key]

    t_start = get_times(ts_pred)[1]
    t_end   = get_times(ts_pred)[end]
    ts_true = select_timespan(ts_true, t_start, t_end)
    ts_true = _check_and_align_locations(ts_true, get_names(ts_pred), "target[\"$out_key\"]")

    if !isnothing(station_names)
        ts_pred = select_locations_by_names(ts_pred, station_names)
        ts_true = select_locations_by_names(ts_true, station_names)
    end
    if !isnothing(timerange)
        ts_pred = select_timespan(ts_pred, timerange[1], timerange[2])
        ts_true = select_timespan(ts_true, timerange[1], timerange[2])
    end

    # Per-station RMSE for the title (matches `_plot_station_series`)
    errors = get_values(ts_true) .- get_values(ts_pred)
    rmses  = sqrt.(mean(abs2, errors; dims=2))[:, 1]

    names = get_names(ts_pred)
    for (i, station) in enumerate(names)
        rmse_str = @sprintf("%.4f", rmses[i])
        p = scatter(ts_true, ts_pred; location_index=i,
                    title="$station  RMSE = $rmse_str")
        add_fit && linear_fit!(p, ts_true, ts_pred; location_index=i)
        add_qq  && qq!(p, ts_true, ts_pred; location_index=i)
        savefig(p, joinpath(save_dir, "$(station).png"))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Per-station statistics
# ──────────────────────────────────────────────────────────────────────────────

"""
    _write_station_stats(output, target, path;
                         timerange=nothing, station_names=nothing)

Internal helper used by `write_outputs`.  Computes per-station validation
statistics via `compute_statistics` and writes them to a CSV file at `path`.
The parent directory must already exist.
"""
function _write_station_stats(output::Dict{String, TimeSeries},
                               target::Dict{String, TimeSeries},
                               path::String;
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
    else
        ts_pred = select_locations_by_names(ts_pred, get_names(ts_true))
    end
    if !isnothing(timerange)
        ts_pred = select_timespan(ts_pred, timerange[1], timerange[2])
        ts_true = select_timespan(ts_true, timerange[1], timerange[2])
    end

    df = compute_statistics(ts_true, ts_pred)
    CSV.write(path, df)
end

# ──────────────────────────────────────────────────────────────────────────────
# Series output
# ──────────────────────────────────────────────────────────────────────────────

"""
    _write_station_series(output, target, save_dir, name, format;
                          timerange=nothing, station_names=nothing)

Internal helper used by `write_outputs`.  Writes the predicted time series to
`save_dir` using `format` (`"netcdf"`, `"jld2"`, or `"noos"`).

- `"netcdf"` / `"jld2"`: single file `series_<name>.<ext>` in `save_dir`.
- `"noos"`: one file per station inside `series_<name>/`.

Existing files are overwritten.
"""
function _write_station_series(output::Dict{String, TimeSeries},
                                target::Dict{String, TimeSeries},
                                save_dir::String,
                                name::String,
                                format::String;
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
    end
    if !isnothing(timerange)
        ts_pred = select_timespan(ts_pred, timerange[1], timerange[2])
    end

    if format == "netcdf"
        path = joinpath(save_dir, "series_$(name).nc")
        isfile(path) && rm(path)
        write_to_netcdf(ts_pred, path)
    elseif format == "jld2"
        path = joinpath(save_dir, "series_$(name).jld2")
        isfile(path) && rm(path)
        write_to_jld2(ts_pred, path)
    elseif format == "noos"
        subdir = joinpath(save_dir, "series_$(name)")
        isdir(subdir) && rm(subdir; recursive=true)
        mkpath(subdir)
        for (i, station) in enumerate(get_names(ts_pred))
            write_single_noos_file(joinpath(subdir, "$(station).noos"), ts_pred, i)
        end
    else
        error("Unknown series_format: \"$format\". Use \"netcdf\", \"jld2\", or \"noos\".")
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Tidal analysis plot helper
# ──────────────────────────────────────────────────────────────────────────────

"""
    _plot_station_tidal_analysis(output, target, save_dir;
                                 const_list=nothing, max_constituents=20,
                                 timerange=nothing, station_names=nothing)

Internal helper used by `write_outputs` for tide models.  Runs harmonic analysis
on both the observed target and the model predictions, then saves a 2-panel
comparison plot per station to `save_dir`:
- Panel 1: amplitude bar chart (observations and predicted side by side).
- Panel 2: phase scatter (observations and predicted overlaid).

`const_list` is a `Vector{String}` of constituent names.  Defaults to
`constituent_list("year")`.  Pass a shorter list (e.g. `constituent_list("month")`)
for short time series.

Failures in `analysis` (e.g. time series too short) are caught and logged so
that the remaining outputs are not interrupted.
"""
function _plot_station_tidal_analysis(output::Dict{String, TimeSeries},
                                       target::Dict{String, TimeSeries},
                                       save_dir::String;
                                       const_list    = nothing,
                                       max_constituents::Integer = 20,
                                       timerange     = nothing,
                                       station_names = nothing)

    isnothing(const_list) && (const_list = constituent_list("year"))

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

    tc_obs  = try
        analysis(ts_true, const_list)
    catch e
        @warn "Tidal analysis failed for observations: $e"
        return
    end
    tc_pred = try
        analysis(ts_pred, const_list)
    catch e
        @warn "Tidal analysis failed for predictions: $e"
        return
    end

    names = get_names(ts_pred)
    for (i, station) in enumerate(names)
        p = plot(tc_obs, tc_pred;
                 location_index  = i,
                 label_ref       = "Observations",
                 label_comp      = "Predicted",
                 max_constituents = max_constituents,
                 size            = (900, 600),
        )
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
