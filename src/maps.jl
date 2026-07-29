# maps.jl
#
# Geographic maps of per-station statistics.  Two layers:
#   * plot_map(bbox; …)            — composable basemap primitive (Layer 1); returns
#                                    a Plots.Plot with the EMODnet bathymetry drawn,
#                                    ready to `scatter!(p, lon, lat; zcolor=…)` onto.
#   * _plot_station_stats(…)       — the `plot_stats` output (Layer 2): per-station
#                                    RMSE & bias maps.
#
# Basemap: EMODnet bathymetry via a direct WMS GetMap (EPSG:4326), fetched once and
# cached to disk (keyed on the full request), with a graceful offline fallback.

using Plots
using FileIO: load
using Printf: @sprintf
import Downloads

const EMODNET_WMS = "https://ows.emodnet-bathymetry.eu/wms"

# ── basemap cache + fetch ─────────────────────────────────────────────────────────

"""On-disk cache directory for basemap images (repo `basemaps/`)."""
_basemap_cache_dir() = joinpath(pkgdir(@__MODULE__), "basemaps")

"""
    _bbox_from_points(lon, lat; margin=0.06) -> (minlon, maxlon, minlat, maxlat)

Bounding box covering all points, padded by `margin` (fraction of each span).
"""
function _bbox_from_points(lon, lat; margin=0.06)
    lo1, lo2 = extrema(lon); la1, la2 = extrema(lat)
    dx = max(lo2 - lo1, 1e-3) * margin; dy = max(la2 - la1, 1e-3) * margin
    return (lo1 - dx, lo2 + dx, la1 - dy, la2 + dy)
end

"""Pixel size `(width, height)` for `bbox` at `ppd` pixels/degree, capped at `maxpx`."""
function _basemap_pixels(bbox; ppd=40, maxpx=1400)
    minlon, maxlon, minlat, maxlat = bbox
    w = clamp(round(Int, (maxlon - minlon) * ppd), 200, maxpx)
    h = clamp(round(Int, (maxlat - minlat) * ppd), 200, maxpx)
    return w, h
end

"""
    _basemap(bbox; layer, cache_dir, width, height, fetch=true) -> String | nothing

Path to a cached EMODnet WMS PNG for the request, fetching + caching it if absent.
The cache filename encodes the full request (layer + bbox + size) so different
backgrounds never collide.  Returns `nothing` (graceful) when `fetch=false` and it
is not cached, or when the download fails.
"""
function _basemap(bbox; layer="mean_atlas_land", cache_dir=_basemap_cache_dir(),
                  width, height, fetch=true)
    minlon, maxlon, minlat, maxlat = bbox
    fname = @sprintf("emodnet-%s__%.4f_%.4f_%.4f_%.4f__%dx%d.png",
                     layer, minlon, minlat, maxlon, maxlat, width, height)
    path = joinpath(cache_dir, fname)
    isfile(path) && return path
    fetch || return nothing
    url = string(EMODNET_WMS,
        "?service=WMS&version=1.1.1&request=GetMap&layers=emodnet:", layer,
        "&srs=EPSG:4326&bbox=", minlon, ",", minlat, ",", maxlon, ",", maxlat,
        "&width=", width, "&height=", height, "&format=image/png&styles=")
    try
        mkpath(cache_dir)
        tmp = tempname() * ".png"
        Downloads.download(url, tmp)
        if filesize(tmp) < 1000          # error XML / empty, not a real image
            rm(tmp; force=true); return nothing
        end
        mv(tmp, path; force=true)
        return path
    catch err
        @warn "basemap fetch failed; plotting without background" exception=err
        return nothing
    end
end

# ── Layer 1: plot_map — composable basemap primitive ──────────────────────────────

"""
    plot_map(bbox; layer="mean_atlas_land", fetch=true, title="", size=(900,860),
             cache_dir=_basemap_cache_dir(), kwargs...) -> Plots.Plot

Return a `Plots.Plot` with the (cached) EMODnet bathymetry basemap drawn on plain
lon/lat axes, ready to `scatter!(p, lon, lat; zcolor=…)` onto.  `bbox` is
`(minlon, maxlon, minlat, maxlat)`.  When the basemap is unavailable (offline and
not cached), returns empty lon/lat axes over `bbox` — the station overlay still
renders.
"""
function plot_map(bbox; layer="mean_atlas_land", fetch=true, title="",
                  size=nothing, cache_dir=_basemap_cache_dir(), kwargs...)
    minlon, maxlon, minlat, maxlat = bbox
    if size === nothing                          # canvas aspect follows the bbox (no letterboxing)
        cw = 900
        ch = clamp(round(Int, cw * (maxlat - minlat) / max(maxlon - minlon, 1e-6)) + 90, 320, 1200)
        size = (cw, ch)
    end
    w, h = _basemap_pixels(bbox)
    path = _basemap(bbox; layer, cache_dir, width=w, height=h, fetch)
    common = (; xlabel="longitude", ylabel="latitude", title=title, size=size,
              framestyle=:box, legend=false)
    path === nothing && return plot(; xlims=(minlon, maxlon), ylims=(minlat, maxlat),
                                    common..., kwargs...)
    img = reverse(load(path); dims=1)            # PNG top row = maxLat -> put minLat first
    xr = range(minlon, maxlon; length=Base.size(img, 2))
    yr = range(minlat, maxlat; length=Base.size(img, 1))
    return plot(xr, yr, img; yflip=false, common..., kwargs...)
end

# ── Layer 2: per-station stat maps + charts (`plot_stats` output) ─────────────────

"""
    _plot_station_stats(output, target, save_dir; stats=[:rmse,:bias],
                        timerange=nothing, station_names=nothing, fetch=true)

Internal helper used by `write_outputs`.  Computes per-station statistics
(`compute_statistics`) and, for each requested stat, saves a map of the stations on
the EMODnet bathymetry background coloured by that stat (`map_<stat>.png`).
`save_dir` must already exist.
"""
function _plot_station_stats(output::Dict{String, TimeSeries},
                             target::Dict{String, TimeSeries},
                             save_dir::String;
                             stats         = [:rmse, :bias],
                             timerange     = nothing,
                             station_names = nothing,
                             fetch::Bool   = true)
    out_key = first(keys(output))
    ts_pred = output[out_key]
    ts_true = target[out_key]

    t_start = get_times(ts_pred)[1]; t_end = get_times(ts_pred)[end]
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

    df   = compute_statistics(ts_true, ts_pred)
    lon  = Float64.(get_longitudes(ts_pred)); lat = Float64.(get_latitudes(ts_pred))
    bbox = _bbox_from_points(lon, lat)

    for s in stats
        s in propertynames(df) || error(
            "_plot_station_stats: unknown stat $(repr(s)); available: $(propertynames(df))")
        vals = Float64.(df[!, s])
        p = plot_map(bbox; title="per-station $(s)", fetch=fetch)
        if s === :bias                       # diverging, centred on 0
            m = maximum(abs, vals); m = m == 0 ? 1.0 : m
            scatter!(p, lon, lat; zcolor=vals, c=:RdBu, clims=(-m, m), markersize=5,
                     markerstrokewidth=0.4, markerstrokecolor=:black,
                     colorbar=true, colorbar_title=string(s))
        else                                 # sequential
            scatter!(p, lon, lat; zcolor=vals, c=:viridis, markersize=5,
                     markerstrokewidth=0.4, markerstrokecolor=:black,
                     colorbar=true, colorbar_title=string(s))
        end
        savefig(p, joinpath(save_dir, "map_$(s).png"))
    end
    return nothing
end
