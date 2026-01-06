# series.jl
#
# Data type for multivariate time series
# and methods to manipulate them.

struct TimeSeries <: AbstractTimeSeries
    values::Matrix{Float32} # values matrix with rows as stations and columns as time #NOTE: I would probably swap indices a next time
    times::Vector{DateTime} # vector of DateTime objects
    names::Vector{String} # vector of station names
    longitudes::Vector{Float64} # vector of longitudes
    latitudes::Vector{Float64} # vector of latitudes
    quantity::String # physical quantity measured (e.g., "water level")
    source::String # source of the data
end

# Constructor from an existing (Abstract)TimeSeries object 
"""
function TimeSeries(values::Matrix{Float32}, times::Vector{DateTime}, names::Vector{String}, longitudes::Vector{Float64}, latitudes::Vector{Float64}, quantity::String, source::String)
Creates a TimeSeries object from the provided values and metadata.
use: ts = TimeSeries(values, times, names, longitudes, latitudes, "water level", "source")
where values is a matrix of Float32 values, times is a vector of DateTime objects, names is a vector of station names, longitudes and latitudes are vectors
of geographic coordinates, quantity is a string describing the physical quantity measured, and source is a string describing the source of the data.
Returns a TimeSeries object containing the provided data and metadata.

A Timeseries object can also be created from an AbstractTimeSeries object using the constructor:
use: ts = TimeSeries(abstract_ts)
where abstract_ts is an AbstractTimeSeries object, eg created from a NetCDF file.
"""
function TimeSeries(ts::AbstractTimeSeries)
    return TimeSeries(get_values(ts), get_times(ts), get_names(ts), get_longitudes(ts), get_latitudes(ts), get_quantity(ts), get_source(ts))
end

#
# Getters for the fields
#
get_values(ts::TimeSeries) = ts.values
get_times(ts::TimeSeries) = ts.times
get_names(ts::TimeSeries) = ts.names
get_longitudes(ts::TimeSeries) = ts.longitudes
get_latitudes(ts::TimeSeries) = ts.latitudes
get_quantity(ts::TimeSeries) = ts.quantity
get_source(ts::TimeSeries) = ts.source

#
# Selection methods - using getters
#  These are the default implementations for the selection methods. 
#  They can be overridden in the specific time series implementations if needed. 
#  Especially for NetCDF and Zarr time series, where a lazy reading method can be applied for large datasets.
#
function select_locations_by_ids(ts::AbstractTimeSeries, location_indices::Vector{T} where T<:Integer)
    selected_values = get_values(ts)[location_indices, :]
    selected_names = get_names(ts)[location_indices]
    selected_longitudes = get_longitudes(ts)[location_indices]
    selected_latitudes = get_latitudes(ts)[location_indices]
    selected_times = get_times(ts)
    selected_quantity = get_quantity(ts)
    selected_source = get_source(ts)
    return TimeSeries(selected_values, selected_times, selected_names, selected_longitudes, selected_latitudes,
                      selected_quantity, selected_source)
end

function select_location_by_id(ts::AbstractTimeSeries, location_index::Integer)
    return select_locations_by_ids(ts, [location_index])
end

function select_locations_by_names(ts::AbstractTimeSeries, location_names::Vector{String})
    all_location_names = get_names(ts)
    location_indices = find_location_indices(location_names, all_location_names)
    return select_locations_by_ids(ts, location_indices)
end

function select_location_by_name(ts::AbstractTimeSeries, location_name::String)
    index = find_location_index(location_name, get_names(ts))
    if index < 0
        error("Location $(location_name) not found in the time series.")
    end
    return select_location_by_id(ts, index)
end

function select_timespan(ts::AbstractTimeSeries, start_time::DateTime, end_time::DateTime)
    times = get_times(ts)
    time_indices = findall(t -> t >= start_time && t <= end_time, times)
    if isempty(time_indices)
        error("No time steps found in the specified timespan.")
    end
    selected_values = get_values(ts)[:, time_indices]
    selected_names = get_names(ts)
    selected_longitudes = get_longitudes(ts)
    selected_latitudes = get_latitudes(ts)
    selected_quantity = get_quantity(ts)
    selected_source = get_source(ts)
    return TimeSeries(selected_values, times[time_indices], selected_names, selected_longitudes, selected_latitudes,
                      selected_quantity, selected_source)
end

function select_timerange_with_fill(ts::AbstractTimeSeries, time_range::StepRange{DateTime, <:TimePeriod}; fill_value=nothing)
    # Collect metadata
    selected_names = get_names(ts)
    selected_longitudes = get_longitudes(ts)
    selected_latitudes = get_latitudes(ts)
    selected_quantity = get_quantity(ts)
    selected_source = get_source(ts)
    # get times and values
    all_times = get_times(ts)
    all_values = get_values(ts)
    # reserve space for selected values and times
    selected_times=collect(time_range)
    n_stations=length(selected_names)
    n_times=length(selected_times)
    T = eltype(all_values)
    selected_values = fill(fill_value === nothing ? convert(T, NaN) : fill_value, n_stations, n_times)
    # loop over time and fill values if available
    for (i, t) in enumerate(selected_times)
        index = findfirst(==(t), all_times)
        if index !== nothing
            selected_values[:, i] = all_values[:, index]
        end
    end
    return TimeSeries(selected_values, selected_times, selected_names, selected_longitudes, selected_latitudes, selected_quantity, selected_source)
end

function select_times_by_ids(ts::AbstractTimeSeries, time_indices::Vector{T} where T<:Integer)
    if isempty(time_indices)
        error("No time indices provided.")
    end
    selected_values = get_values(ts)[:, time_indices] # Default implementation reads all values for the selected times
    selected_names = get_names(ts)
    selected_longitudes = get_longitudes(ts)
    selected_latitudes = get_latitudes(ts)
    selected_times = get_times(ts)[time_indices]
    selected_quantity = get_quantity(ts)
    selected_source = get_source(ts)
    return TimeSeries(selected_values, selected_times, selected_names, selected_longitudes, selected_latitudes,
                      selected_quantity, selected_source)
end

function select_times_by_ids(ts::TimeSeries, time_indices::Vector{T} where T<:Integer)
    if isempty(time_indices)
        error("No time indices provided.")
    end
    selected_names = get_names(ts)
    selected_longitudes = get_longitudes(ts)
    selected_latitudes = get_latitudes(ts)
    selected_times = get_times(ts)[time_indices]
    selected_quantity = get_quantity(ts)
    selected_source = get_source(ts)
    # Read the values for the selected times
    values = ts.values
    n_stations,n_times = size(values)
    selected_values = zeros(Float32, n_stations, length(time_indices)) # Preallocate for selected values
    # Read the values for the selected times
    for (i, t) in enumerate(time_indices)
        selected_values[:, i] = values[:, t]
    end

    # Create a new TimeSeries object with the selected values
    return TimeSeries(selected_values, selected_times, selected_names, selected_longitudes, selected_latitudes,
                      selected_quantity, selected_source)
end

function merge_by_times(ts1::AbstractTimeSeries, ts2::AbstractTimeSeries)
    # Take metadata from ts1, and fill gaps with ts2
    quantity = get_quantity(ts1)
    if quantity==""
        quantity = get_quantity(ts2)
    end
    source= get_source(ts1)
    if source==""
        source = get_source(ts2)
    end
    # Check for an equal number of locations and same names
    if length(get_names(ts1)) != length(get_names(ts2))
        error("Cannot merge time series with different number of locations.")
    end
    names=get_names(ts1)
    names2=get_names(ts2)
    for name in names
        if !(name in names2)
            error("Cannot merge time series with different location names. Location $(name) not found in second time series.")
        end
    end
    # Merge times and values
    times1=get_times(ts1)
    times2=get_times(ts2)
    start_index=findfirst(t -> t > times1[end], times2)
    if start_index === nothing # the second time series ends before the first one, swap them
        return merge_by_time(ts2, ts1)
    end
    all_times = vcat(times1, times2[start_index:end])
    # Merge values, find indices of names in ts2 to account for different order of names
    values2=get_values(ts2)
    sorted_values2 = similar(values2)
    for (index1,name) in enumerate(names)
        index2=findfirst(==(name), names2)
        sorted_values2[index1,:] .= values2[index2,:]
    end
    all_values = hcat(get_values(ts1), sorted_values2[:, start_index:end])
    return TimeSeries(all_values, all_times, names, get_longitudes(ts1), get_latitudes(ts1),quantity, source)
end 

"""
 unction merge_by_locations(series_vector::Vector{AbstractTimeSeries})

 Merge a vector of TimeSeries (or AbstractTimeSeries)
 by location name. Each TimeSeries is assumed to contain one location only. The source and quatity must be the same. The times must match too.
"""
function merge_by_locations(series_vector::Vector{<:AbstractTimeSeries})
    if length(series_vector)<=0
        return nothing
    end
    source=get_source(series_vector[1])
    quantity=get_quantity(series_vector[1])
    times=copy(get_times(series_vector[1]))
    n_locations=length(series_vector)
    names=Vector{String}()
    latitudes=Vector{Float64}()
    longitudes=Vector{Float64}()
    values=zeros(Float32,n_locations,length(times))
    for iloc=1:n_locations
        this_series=series_vector[iloc]
        push!(names,get_names(this_series)[1])
        push!(longitudes,get_longitudes(this_series)[1])
        push!(latitudes,get_latitudes(this_series)[1])
        if get_quantity(this_series) != quantity
            error("Cannot merge time series with different quantities: $(get_quantity(this_series)) != $quantity")
        end
        if get_source(this_series) != source
            error("Cannot merge time series with different sources: $(get_source(this_series)) != $source")
        end 
        values[iloc,:]=get_values(this_series)[1,:]
    end
    return TimeSeries(values, times, names, longitudes, latitudes, quantity, source)
end

# Show function for TimeSeries
function Base.show(io::IO, series::AbstractTimeSeries)
    println(io, "AbstractTimeSeries:")
    println(io, "   Quantity: ", get_quantity(series))
    println(io, "   Source: ", get_source(series))
    println(io, "   Number of locations: ", length(get_names(series)))
    println(io, "   Number of time points: ", length(get_times(series)))
    println(io, "   Data shape: ", size(get_values(series)))
    println(io, "   Times: ", get_times(series)[1], " to ", get_times(series)[end])
    println(io, "   Locations: ", join(get_names(series), ", "))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", series::AbstractTimeSeries)
    println(io, "AbstractTimeSeries: $(get_quantity(series)) from $(get_source(series)), with $(length(get_names(series))) locations, from $(get_times(series)[1]) until $(get_times(series)[end]).")
    #show(io, series)
    return nothing
end

function Base.show(io::IO, series::TimeSeries)
    println(io, "TimeSeries:")
    println(io, "   Quantity: ", get_quantity(series))
    println(io, "   Source: ", get_source(series))
    println(io, "   Number of locations: ", length(get_names(series)))
    println(io, "   Number of time points: ", length(get_times(series)))
    println(io, "   Data shape: ", size(get_values(series)))
    println(io, "   Times: ", get_times(series)[1], " to ", get_times(series)[end])
    println(io, "   Locations: ", join(get_names(series), ", "))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", series::TimeSeries)
    println(io, "TimeSeries: $(get_quantity(series)) from $(get_source(series)), with $(length(get_names(series))) locations, from $(get_times(series)[1]) until $(get_times(series)[end]).")
    #show(io, series)
    return nothing
end