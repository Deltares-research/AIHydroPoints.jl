# constituents.jl
#
# Data type for tidal constituents across multiple locations.
# Analogous to series.jl, with the following mapping:
#
#   TimeSeries                       TidalConstituents
#   ─────────────────────────────    ────────────────────────────────────────
#   values[locations × times]     →  amplitudes[locations × constituents]
#                                    phases[locations × constituents]
#   times::Vector{DateTime}       →  constituent_names::Vector{String}
#   names, lons, lats, qty, src   →  names, lons, lats, qty, src  (identical)
#
# All locations share the same set of constituents: either all have M4 or none do.

# Extend MultiTimeSeries generic functions so that TidalConstituents shares the same
# interface as TimeSeries (single-dispatch, no ambiguity).
import MultiTimeSeries: get_names, get_longitudes, get_latitudes, get_quantity, get_source
import MultiTimeSeries: select_locations_by_ids, select_location_by_id
import MultiTimeSeries: select_locations_by_names, select_location_by_name

struct TidalConstituents <: AbstractTidalConstituents
    amplitudes::Matrix{Float32}        # amplitudes in metres [locations × constituents]
    phases::Matrix{Float32}            # phases in degrees [locations × constituents]
    constituent_names::Vector{String}  # constituent names (e.g. ["M2", "S2", "M4"])
    names::Vector{String}              # station names
    longitudes::Vector{Float64}        # longitudes of stations
    latitudes::Vector{Float64}         # latitudes of stations
    quantity::String                   # physical quantity (e.g. "water level")
    source::String                     # source of the data or analysis
end

"""
    TidalConstituents(tc::AbstractTidalConstituents)

Copy constructor from any AbstractTidalConstituents object.
"""
function TidalConstituents(tc::AbstractTidalConstituents)
    return TidalConstituents(
        get_amplitudes(tc), get_phases(tc), get_constituent_names(tc),
        get_names(tc), get_longitudes(tc), get_latitudes(tc),
        get_quantity(tc), get_source(tc),
    )
end

#
# Getters
#
get_amplitudes(tc::TidalConstituents)       = tc.amplitudes
get_phases(tc::TidalConstituents)           = tc.phases
get_constituent_names(tc::TidalConstituents) = tc.constituent_names
get_names(tc::TidalConstituents)            = tc.names
get_longitudes(tc::TidalConstituents)       = tc.longitudes
get_latitudes(tc::TidalConstituents)        = tc.latitudes
get_quantity(tc::TidalConstituents)         = tc.quantity
get_source(tc::TidalConstituents)           = tc.source

#
# Selection by location — analogous to select_locations_by_ids / select_location_by_name in series.jl
#
function select_locations_by_ids(tc::AbstractTidalConstituents, location_indices::Vector{T} where T<:Integer)
    return TidalConstituents(
        get_amplitudes(tc)[location_indices, :],
        get_phases(tc)[location_indices, :],
        get_constituent_names(tc),
        get_names(tc)[location_indices],
        get_longitudes(tc)[location_indices],
        get_latitudes(tc)[location_indices],
        get_quantity(tc),
        get_source(tc),
    )
end

function select_location_by_id(tc::AbstractTidalConstituents, location_index::Integer)
    return select_locations_by_ids(tc, [location_index])
end

function select_locations_by_names(tc::AbstractTidalConstituents, location_names::Vector{String})
    all_names = get_names(tc)
    indices = [findfirst(==(n), all_names) for n in location_names]
    missing_names = location_names[findall(isnothing, indices)]
    if !isempty(missing_names)
        error("Locations not found in TidalConstituents: " * join(missing_names, ", "))
    end
    return select_locations_by_ids(tc, Int.(indices))
end

function select_location_by_name(tc::AbstractTidalConstituents, location_name::String)
    return select_locations_by_names(tc, [location_name])
end

#
# Selection by constituent — analogous to select_timespan in series.jl
#
"""
    select_constituents_by_names(tc, names) -> TidalConstituents

Return a new TidalConstituents containing only the requested constituents.
All locations are kept; the shared constituent list is reduced to `names`.
"""
function select_constituents_by_names(tc::AbstractTidalConstituents, names::Vector{String})
    all_consts = get_constituent_names(tc)
    indices = [findfirst(==(n), all_consts) for n in names]
    missing_consts = names[findall(isnothing, indices)]
    if !isempty(missing_consts)
        error("Constituents not found in TidalConstituents: " * join(missing_consts, ", "))
    end
    idx = Int.(indices)
    return TidalConstituents(
        get_amplitudes(tc)[:, idx],
        get_phases(tc)[:, idx],
        names,
        get_names(tc),
        get_longitudes(tc),
        get_latitudes(tc),
        get_quantity(tc),
        get_source(tc),
    )
end

#
# Merge by locations — analogous to merge_by_locations in series.jl
# Each element of the vector must share the same constituent set.
#
"""
    merge_by_locations(constituents_vector) -> TidalConstituents

Merge a vector of TidalConstituents (one location each) into a single object.
All elements must share the same constituent names, quantity, and source.
"""
function merge_by_locations(constituents_vector::Vector{<:AbstractTidalConstituents})
    if isempty(constituents_vector)
        error("Cannot merge an empty vector of TidalConstituents.")
    end
    ref = constituents_vector[1]
    quantity        = get_quantity(ref)
    source          = get_source(ref)
    constituent_names = get_constituent_names(ref)
    n_consts        = length(constituent_names)
    n_locations     = length(constituents_vector)
    names           = Vector{String}()
    longitudes      = Vector{Float64}()
    latitudes       = Vector{Float64}()
    amplitudes      = zeros(Float32, n_locations, n_consts)
    phases          = zeros(Float32, n_locations, n_consts)
    for (i, tc) in enumerate(constituents_vector)
        if get_constituent_names(tc) != constituent_names
            error("Cannot merge TidalConstituents with different constituent sets.")
        end
        if get_quantity(tc) != quantity
            error("Cannot merge TidalConstituents with different quantities: $(get_quantity(tc)) != $quantity")
        end
        if get_source(tc) != source
            error("Cannot merge TidalConstituents with different sources: $(get_source(tc)) != $source")
        end
        push!(names,      get_names(tc)[1])
        push!(longitudes, get_longitudes(tc)[1])
        push!(latitudes,  get_latitudes(tc)[1])
        amplitudes[i, :] = get_amplitudes(tc)[1, :]
        phases[i, :]     = get_phases(tc)[1, :]
    end
    return TidalConstituents(amplitudes, phases, constituent_names,
                             names, longitudes, latitudes, quantity, source)
end

#
# Show — analogous to Base.show in series.jl
#
function Base.show(io::IO, tc::AbstractTidalConstituents)
    println(io, "AbstractTidalConstituents:")
    println(io, "   Quantity:         ", get_quantity(tc))
    println(io, "   Source:           ", get_source(tc))
    println(io, "   Locations:        ", join(get_names(tc), ", "))
    println(io, "   N locations:      ", length(get_names(tc)))
    println(io, "   N constituents:   ", length(get_constituent_names(tc)))
    println(io, "   Constituents:     ", join(get_constituent_names(tc), ", "))
    println(io, "   Data shape:       ", size(get_amplitudes(tc)), " [locations × constituents]")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", tc::AbstractTidalConstituents)
    println(io, "AbstractTidalConstituents: $(get_quantity(tc)) from $(get_source(tc)), " *
                "$(length(get_names(tc))) locations, $(length(get_constituent_names(tc))) constituents.")
    return nothing
end

function Base.show(io::IO, tc::TidalConstituents)
    println(io, "TidalConstituents:")
    println(io, "   Quantity:         ", get_quantity(tc))
    println(io, "   Source:           ", get_source(tc))
    println(io, "   Locations:        ", join(get_names(tc), ", "))
    println(io, "   N locations:      ", length(get_names(tc)))
    println(io, "   N constituents:   ", length(get_constituent_names(tc)))
    println(io, "   Constituents:     ", join(get_constituent_names(tc), ", "))
    println(io, "   Data shape:       ", size(get_amplitudes(tc)), " [locations × constituents]")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", tc::TidalConstituents)
    println(io, "TidalConstituents: $(get_quantity(tc)) from $(get_source(tc)), " *
                "$(length(get_names(tc))) locations, $(length(get_constituent_names(tc))) constituents.")
    return nothing
end
