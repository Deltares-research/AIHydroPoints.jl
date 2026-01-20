# abstract_series.jl
#
# Abstract datatype for time series at multiple locations.

"""
 Abstract datatype for time series at multiple locations.
    The type conceptually contains:
        values::Matrix{Float32} # values matrix with rows as stations and columns as time
        times::Vector{DateTime} # vector of DateTime objects
        names::Vector{String} # vector of station names
        longitudes::Vector{Float64} # vector of longitudes
        latitudes::Vector{Float64} # vector of latitudes
        quantity::String # physical quantity measured (e.g., "water level")
        source::String # source of the values
    But the implementation is abstract and does not contain these fields.
"""
abstract type AbstractTimeSeries 
end

#
# empty getters for the fields
#
get_values(ts::AbstractTimeSeries) = error("get_values not implemented for AbstractTimeSeries")
get_times(ts::AbstractTimeSeries) = error("get_times not implemented for AbstractTimeSeries")
get_names(ts::AbstractTimeSeries) = error("get_names not implemented for AbstractTimeSeries")
get_longitudes(ts::AbstractTimeSeries) = error("get_longitudes not implemented for AbstractTimeSeries")
get_latitudes(ts::AbstractTimeSeries) = error("get_latitudes not implemented for AbstractTimeSeries")
get_quantity(ts::AbstractTimeSeries) = error("get_quantity not implemented for AbstractTimeSeries")
get_source(ts::AbstractTimeSeries) = error("get_source not implemented for AbstractTimeSeries")

# #
# # Selection methods
# #
# There are no default implementations for the selection methods in the abstract type in series.jl.
#
# select_locations_by_ids(ts::AbstractTimeSeries, location_indices::Vector{T} where T<:Integer) = error("select_locations_by_ids not implemented for AbstractTimeSeries")
# select_location_by_id(ts::AbstractTimeSeries, location_index::Integer) = error("select_location_by_id not implemented for AbstractTimeSeries")
# select_locations_by_names(ts::AbstractTimeSeries, location_names::Vector{String}) = error("select_locations_by_names not implemented for AbstractTimeSeries")
# select_location_by_name(ts::AbstractTimeSeries, location_name::String) = error("select_location_by_name not implemented for AbstractTimeSeries")
# select_timespan(ts::AbstractTimeSeries, start_time::DateTime, end_time::DateTime) = error("select_timespan not implemented for AbstractTimeSeries")

#
# selection utilities
#

# Find the index of a location by name
"""
function find_location_index(location_name::String, location_names::Vector{String})
Finds the index of a location in a vector of Strings.
use: index = find_location_index("Station A",["Station A", "Station B", "Station C"])
where location_name is the name of the location to find and location_names is a vector of location names.
Returns the index of the location in the series.
If the location is not found, it returns -1.
"""
function find_location_index(location_name::String, location_names::Vector{String})
    result=findfirst(x -> x == location_name, location_names)
    if isnothing(result)
        return -1
    else
        return result
    end
end

# Collect station indices by name
"""
function find_location_indices(location_selection::Vector{String}, all_location_names::Vector{String})
Collects the indices of stations in a list of location names.
use: indices = collect_station_indices(["Station A", "Station C","Station D"], ["Station A", "Station B", "Station C"])
 results in indices = [1, 3, -1]
Returns a vector of indices corresponding to the station names.
If a station name is not found, then a -1 is returned for that station.
"""
function find_location_index(location_selection::Vector{String},all_location_names::Vector{String})
    indices = Vector{Int}()
    for name in location_selection
        index = find_location_index(name, all_location_names)
        push!(indices, index)
    end
    return indices
end

# function find_time_index(time::DateTime, all_times::Vector{DateTime})
#     result = findfirst(x->x==time, all_times)
#     if isnothing(result)
#         return -1
#     else
#         return result   
#     end
# end

# function find_time_index(times::Vector{DateTime}, all_times::Vector{DateTime})
#     indices = Vector{Int}()
#     for time in times
#         index = find_time_index(time, all_times)
#         push!(indices, index)
#     end
#     return indices
# end