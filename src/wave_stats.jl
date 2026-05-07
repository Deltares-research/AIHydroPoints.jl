using Statistics
using DataFrames

"""
    stats_skipnan(y_true::TimeSeries, y_pred::TimeSeries) -> DataFrame

Compute per-station statistics (bias, RMSE, MAE, relative bias, scatter index),
skipping NaN values.  Returns a `DataFrame` with one row per station.
"""
function stats_skipnan(y_true::TimeSeries, y_pred::TimeSeries)
    @assert get_times(y_true) == get_times(y_pred) "Times of true and predicted series differ"
    @assert get_names(y_true) == get_names(y_pred) "Station names of true and predicted series differ"

    names        = get_names(y_true)
    y_true_vals  = copy(Float32.(get_values(y_true)))
    y_pred_vals  = Float32.(get_values(y_pred))
    res          = y_pred_vals .- y_true_vals
    count_notnan = sum(.!isnan.(res), dims=2)

    res[isnan.(res)]                .= 0.0f0
    y_true_vals[isnan.(y_true_vals)] .= 0.0f0
    rel_res = res ./ max.(y_true_vals, 0.1f0)

    bias          = sum(res,           dims=2) ./ count_notnan
    rmse          = sqrt.(sum(res.^2,  dims=2) ./ count_notnan)
    mae           = sum(abs.(res),     dims=2) ./ count_notnan
    relative_bias = sum(rel_res,       dims=2) ./ count_notnan
    scatter_index = sqrt.(sum(rel_res.^2, dims=2) ./ count_notnan)

    return DataFrame(
        station_name  = names,
        bias          = vec(bias),
        rmse          = vec(rmse),
        mae           = vec(mae),
        relative_bias = vec(relative_bias),
        scatter_index = vec(scatter_index),
        count         = vec(count_notnan),
    )
end

"""
    average_stats(previous_stats, stats::DataFrame, timespan_name::String) -> DataFrame

Append a row of station-averaged statistics to `previous_stats` (or create a new
DataFrame if `previous_stats === nothing`).
"""
function average_stats(previous_stats, stats::DataFrame, timespan_name::String)
    row = DataFrame(
        timespan          = timespan_name,
        avg_bias          = mean(stats.bias),
        avg_rmse          = mean(stats.rmse),
        avg_mae           = mean(stats.mae),
        avg_relative_bias = mean(stats.relative_bias),
        avg_scatter_index = mean(stats.scatter_index),
        nstations         = nrow(stats),
    )
    return previous_stats === nothing ? row : vcat(previous_stats, row)
end
