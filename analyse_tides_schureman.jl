# analyse_tides_schureman.jl
#
# Harmonic tidal analysis (Schureman method) on DCSM-FM water-level data.
# For each station, fits tidal constituents by least-squares, predicts the tidal
# signal, and computes surge (observed minus predicted).
# Data files (surge, tides) are written to data/.
# Diagnostic files (constituents CSV, statistics CSV, plots) go to output_dir.
#
# Usage:
#   pixi run julia --project analyse_tides_schureman.jl [input_file [output_dir]]
#
# Defaults:
#   input_file = data/DCSM-FM_0_5nm_2000_2022_5stations_his.jld2
#   output_dir = output_tides_schureman

cd(@__DIR__)
using Pkg
Pkg.activate(".")

using AIHydroPoints
using hatyan_core
using CSV
using DataFrames
using Dates
using Plots
using Statistics

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
input_file = length(ARGS) >= 1 ? ARGS[1] : joinpath("data", "DCSM-FM_0_5nm_2000_2022_5stations_his.jld2")
output_dir = length(ARGS) >= 2 ? ARGS[2] : "output_tides_schureman"

# Fixed 1-month zoom window for diagnostic plots (Andrea storm, Jan 2012)
zoom_start = DateTime(2012, 1, 1)
zoom_end   = DateTime(2012, 2, 1)

method     = "schureman"
const_list = constituent_list("year")   # 94 constituents

rm(output_dir, recursive=true, force=true)
mkpath(output_dir)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
@info "Loading $input_file"
ts = JLD2TimeSeries(input_file)

stations  = get_names(ts)
times     = get_times(ts)
nstations = length(stations)
@info "Stations: $stations"
@info "Time span: $(times[1]) → $(times[end])  ($(length(times)) steps)"

# ──────────────────────────────────────────────
# Harmonic analysis (all stations at once)
# ──────────────────────────────────────────────
@info "Performing harmonic analysis ($method, $(length(const_list)) constituents)…"
tc = analysis(ts, const_list, method)

# ──────────────────────────────────────────────
# Tidal prediction
# ──────────────────────────────────────────────
@info "Predicting tides…"
ts_tides = prediction(tc, times)

# ──────────────────────────────────────────────
# Surge = observed − predicted
# ──────────────────────────────────────────────
surge_values = get_values(ts) .- get_values(ts_tides)
ts_surge = TimeSeries(
    surge_values,
    times,
    get_names(ts),
    get_longitudes(ts),
    get_latitudes(ts),
    "surge",
    get_source(ts),
)

# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────
@info "Statistics (predicted tides vs observed):"
stats = compute_statistics(ts, ts_tides)
println(stats)

surge_std_col = [std(surge_values[i, :]) for i in 1:nstations]
stats[!, :surge_std] = surge_std_col
CSV.write(joinpath(output_dir, "statistics_$(method).csv"), stats)
@info "Statistics written to $(output_dir)/statistics_$(method).csv"

# ──────────────────────────────────────────────
# Write tidal constituents (amplitude & phase per station)
# ──────────────────────────────────────────────
const_names = get_constituent_names(tc)
amplitudes  = get_amplitudes(tc)
phases      = get_phases(tc)

tc_df = DataFrame(constituent = const_names)
for (i, station) in enumerate(stations)
    tc_df[!, "amp_$(station)"]   = Float64.(amplitudes[i, :])
    tc_df[!, "phase_$(station)"] = Float64.(phases[i, :])
end
CSV.write(joinpath(output_dir, "constituents_$(method).csv"), tc_df)
@info "Constituents written to $(output_dir)/constituents_$(method).csv"

# ──────────────────────────────────────────────
# Write JLD2 data files to data/
# ──────────────────────────────────────────────
tides_jld2 = joinpath("data", "tides_schureman_2000_2022_5stations.jld2")
surge_jld2 = joinpath("data", "surge_schureman_2000_2022_5stations.jld2")

@info "Writing $tides_jld2"
write_to_jld2(ts_tides, tides_jld2)

@info "Writing $surge_jld2"
write_to_jld2(ts_surge, surge_jld2)

# ──────────────────────────────────────────────
# Plots — fixed 1-month zoom (Jan 2012, Andrea storm)
# ──────────────────────────────────────────────
ts_zoom       = select_timespan(ts,       zoom_start, zoom_end)
ts_tides_zoom = select_timespan(ts_tides, zoom_start, zoom_end)
ts_surge_zoom = select_timespan(ts_surge, zoom_start, zoom_end)
times_zoom    = get_times(ts_zoom)

for (i, station) in enumerate(stations)
    h      = get_values(ts_zoom)[i, :]
    h_tide = get_values(ts_tides_zoom)[i, :]
    h_surv = get_values(ts_surge_zoom)[i, :]

    p1 = plot(times_zoom, h,      label="Observed",  ylabel="Water level (m)",
              title="$station — $(Dates.format(zoom_start, "u yyyy"))")
    plot!(p1, times_zoom, h_tide, label="Tides ($method)")
    p2 = plot(times_zoom, h_surv, label="Surge", ylabel="Surge (m)", xlabel="Time")

    fig = plot(p1, p2, layout=(2, 1), size=(1000, 600))
    savefig(fig, joinpath(output_dir, "$(station)_zoom.png"))
    @info "  Saved zoom plot for $station"
end

@info "Done."
@info "  Data files : $tides_jld2, $surge_jld2"
@info "  Diagnostics: $output_dir/"
