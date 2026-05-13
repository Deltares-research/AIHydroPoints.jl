# analyse_tides_schureman.jl
#
# Harmonic tidal analysis (Schureman method) on DCSM-FM 2010 water-level data.
# For each station, fits tidal constituents by least-squares, predicts the tidal
# signal, and computes the practical surge (observed minus predicted).
# Outputs tides and surge as NetCDF his-files in output_tides/.

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
year       = length(ARGS) >= 1 ? ARGS[1] : "2010"

# input_file = joinpath(@__DIR__, "test_data", "DCSM-FM_0_5nm_$(year)_5stations_his.jld2")
input_file = joinpath(@__DIR__, "data","DCSM-FM_0_5nm_2010_5stations_his.jld2") # temporary check
output_dir = joinpath(@__DIR__, "output_tides_$(year)")
method     = "schureman"

# Constituent set: "year" requires ~1 year of data (A0 + 94 constituents)
const_list = constituent_list("year")

rm(output_dir, recursive=true, force=true)
mkpath(output_dir)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
@info "Loading $input_file"
# ts = JLD2TimeSeries(input_file, varname="waterlevel")
ts = JLD2TimeSeries(input_file,)

@show ts

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
@show tc

# ──────────────────────────────────────────────
# Tidal prediction
# ──────────────────────────────────────────────
@info "Predicting tides…"
ts_tides = prediction(tc, times)
@show ts_tides

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

# Add surge std to the statistics table and save as CSV
surge_std_col = [std(surge_values[i,:]) for i in 1:nstations]
stats[!, :surge_std] = surge_std_col
CSV.write(joinpath(output_dir, "statistics_$(method)_$(year).csv"), stats)
@info "Statistics written to statistics_$(method)_$(year).csv"

# ──────────────────────────────────────────────
# Write tidal constituents (amplitude & phase per station)
# ──────────────────────────────────────────────
const_names = get_constituent_names(tc)
amplitudes  = get_amplitudes(tc)   # (nstations × nconstituents)
phases      = get_phases(tc)       # (nstations × nconstituents)

# One CSV with columns: constituent, then amp_STATION / phase_STATION pairs
tc_df = DataFrame(constituent = const_names)
for (i, station) in enumerate(stations)
    tc_df[!, "amp_$(station)"]   = Float64.(amplitudes[i, :])
    tc_df[!, "phase_$(station)"] = Float64.(phases[i, :])
end
CSV.write(joinpath(output_dir, "constituents_$(method)_$(year).csv"), tc_df)
@info "Constituents written to constituents_$(method)_$(year).csv"

# ──────────────────────────────────────────────
# Write NetCDF output
# ──────────────────────────────────────────────
tides_nc = joinpath(output_dir, "tides_$(method)_$(year).nc")
surge_nc  = joinpath(output_dir, "surge_$(method)_$(year).nc")

@info "Writing $tides_nc"
write_to_netcdf(ts_tides, tides_nc)

@info "Writing $surge_nc"
write_to_netcdf(ts_surge, surge_nc)

# ──────────────────────────────────────────────
# Plots — full year, one per station
# ──────────────────────────────────────────────
for (i, station) in enumerate(stations)
    h      = get_values(ts)[i,:]
    h_tide = get_values(ts_tides)[i,:]
    h_surv = surge_values[i,:]

    p1 = plot(times, h,      label="Observed",  ylabel="Water level (m)", title=station)
    plot!(p1, times, h_tide, label="Tides ($method)")
    p2 = plot(times, h_surv, label="Surge",     ylabel="Surge (m)", xlabel="Time")

    fig = plot(p1, p2, layout=(2,1), size=(1000, 600))
    savefig(fig, joinpath(output_dir, "$(station)_tides_surge.png"))
    @info "  Saved full-year plot for $station"
end

# ──────────────────────────────────────────────
# Plots — January 1–15 zoom, one per station
# ──────────────────────────────────────────────
yr        = parse(Int, year)
jan_start = DateTime(yr, 1, 1)
jan_end   = DateTime(yr, 1, 15)
ts_jan       = select_timespan(ts,       jan_start, jan_end)
ts_tides_jan = select_timespan(ts_tides, jan_start, jan_end)
ts_surge_jan = select_timespan(ts_surge, jan_start, jan_end)
times_jan    = get_times(ts_jan)

for (i, station) in enumerate(stations)
    h      = get_values(ts_jan)[i,:]
    h_tide = get_values(ts_tides_jan)[i,:]
    h_surv = get_values(ts_surge_jan)[i,:]

    p1 = plot(times_jan, h,      label="Observed",  ylabel="Water level (m)",
              title="$station — Jan 1–15 $year")
    plot!(p1, times_jan, h_tide, label="Tides ($method)")
    p2 = plot(times_jan, h_surv, label="Surge",     ylabel="Surge (m)", xlabel="Time")

    fig = plot(p1, p2, layout=(2,1), size=(1000, 600))
    savefig(fig, joinpath(output_dir, "$(station)_tides_surge_jan.png"))
    @info "  Saved January plot for $station"
end

@info "Done. Output written to $output_dir"
