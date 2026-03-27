# series_noos.jl
# Read and write time series in an ascii based format used by NOOS. 
# There is on  time series per file, with a header that contains metadata.
# The time series data is in columns, with the first column being the time in a specified format.
# The other columns are the time series values, one column per location.
#
# Example:
# #------------------------------------------------------
# # Timeseries retrieved from the MATROOS series database
# # Created at Tue Oct  7 14:40:41 CEST 2025
# #------------------------------------------------------
# # Location    : K13a
# # Position    : (3.219036,53.218117)
# # Source      : swan_dcsm_harmonie
# # Unit        : wave_height
# # Analyse time: 202401020300
# # Timezone    : GMT
# #------------------------------------------------------
# 202312101000    2.58000
# 202312101100    2.41800
# 202312101200    2.21300
# 202312101300    2.01900
# 202312101400    1.82900

using Dates
using Printf

struct NoosTimeSeriesCollection
    # collect by source and quantity
    # assumes that for each source and quantity ther the timeseries all have the same times
    series_list::Dict{Tuple{String, String}, TimeSeries}
end

# constructor that reads all .noos files in a directory and its subdirectories
function NoosTimeSeriesCollection(input_dir::String)
    if !isdir(input_dir)
        error("Input directory does not exist: $input_dir")
    end
    filenames = String[]
    for (root, dirs, files) in walkdir(input_dir)
        for file in files
            if endswith(file, ".noos")
                push!(filenames, joinpath(root, file))
            end
        end
    end
    # read and merge by times
    all_series = Dict{Tuple{String,String,String},TimeSeries}()
    first_time=nothing
    last_time=nothing
    dt=Dates.Hour(1) # assume hourly data
    for filename in filenames
        display("Reading file: $filename")
        series = read_single_noos_file(filename)
        quantity=get_quantity(series)
        source=get_source(series)
        name=get_names(series)[1] # only one series per item at this time
        times=get_times(series)
        if first_time===nothing || times[1]<first_time
            first_time=times[1]
        end
        if last_time===nothing || times[end]>last_time
            last_time=times[end]
        end
        # merge times for each source, quantity and name
        if !haskey(all_series,(source,quantity,name))
            all_series[source,quantity,name]=series
        else
            all_series[source,quantity,name]=merge_by_times(all_series[source,quantity,name],series)
        end     
    end
    # Sync all times to the same time vector
    common_times=first_time:dt:last_time
    for (source, quantity, name) in keys(all_series)
        all_series[source,quantity,name]=select_timerange_with_fill(all_series[source,quantity,name],common_times; fill_value=nothing)
    end
    # collect by source and quantity
    collected_series=Dict{Tuple{String,String},Vector{TimeSeries}}()
    for (source, quantity, name) in keys(all_series) #loop over keys of all_series
        if !haskey(collected_series,(source,quantity))
            collected_series[source,quantity]=Vector{TimeSeries}()
        end
        push!(collected_series[source,quantity],all_series[source,quantity,name])
    end
    # create a merged TimeSeries for each source and quantity
    series_collection=Dict{Tuple{String,String},TimeSeries}()
    for (source, quantity) in keys(collected_series)
        series_collection[source,quantity]=merge_by_locations(collected_series[source,quantity])
    end
    # create a NoosTimeSeriesCollection
    return NoosTimeSeriesCollection(series_collection)
end


function get_source_quantity_keys(collection::NoosTimeSeriesCollection)
    return keys(collection.series_list)
end

function get_sources(collection::NoosTimeSeriesCollection)
    source_quantity_keys = get_source_quantity_keys(collection)
    # sources are the first element of the tuples
    sources = Vector{String}()
    for (source, quantity) in source_quantity_keys
        # the source may be in the list already
        if source in sources
            continue
        end
        push!(sources, source)
    end
    return sources
end

function get_quantities(collection::NoosTimeSeriesCollection)
    source_quantity_keys = get_source_quantity_keys(collection)
    # quantities are the second element of the tuples
    quantities = Vector{String}()
    for (source, quantity) in source_quantity_keys
        # the quantity may be in the list already
        if quantity in quantities
            continue
        end
        push!(quantities, quantity)
    end
    return quantities
end

function get_series_from_collection(collection::NoosTimeSeriesCollection, source::String, quantity::String)
    if !haskey(collection.series_list,(source,quantity))
        error("No series found for source: $source and quantity: $quantity")
    end
    return collection.series_list[(source,quantity)]
end

function read_muliple_noos_files(filenames::Vector{String})
    series_list = TimeSeries[]
    for filename in filenames
        series = read_single_noos_file(filename)
        push!(series_list, series)
    end
    return series_list
end

function read_single_noos_file(filename::String)
    # check if file exists
    if !isfile(filename)
        error("File not found: $filename")
    end
    # Read the file
    times = DateTime[]
    values = Float32[]
    name = ""
    longitude = 0.0
    latitude = 0.0
    source = ""
    quantity = ""

    open(filename, "r") do file
        lines = readlines(file)
        # Parse header lines starting with #
        header_lines = filter(line -> startswith(line, "#"), lines)
        time_format = "yyyymmddHHMM" # default format
        for line in header_lines
            if occursin("Location", line)
                parts = split(line, ":")
                if length(parts) == 2
                    name = strip(parts[2])
                end
            elseif occursin("Position", line)
                parts = split(line, ":")
                if length(parts) == 2
                    pos_str = strip(parts[2])
                    pos_parts = split(replace(pos_str, "(" => "", ")" => ""), ",")
                    if length(pos_parts) == 2
                        longitude = parse(Float64, strip(pos_parts[1]))
                        latitude = parse(Float64, strip(pos_parts[2]))
                    end
                end
            elseif occursin("Source", line)
                parts = split(line, ":")
                if length(parts) == 2
                    source = strip(parts[2])
                end
            elseif occursin("Unit", line)
                parts = split(line, ":")
                if length(parts) == 2
                    quantity = strip(parts[2])
                end
            end
        end
        # Filter out comment lines
        data_lines = filter(line -> !startswith(line, "#"), lines)
        # Parse the data lines
        for line in data_lines
            parts = split(strip(line))
            if length(parts) >= 2
                time_str = parts[1]
                value_str = parts[2]
                push!(times, DateTime(time_str, time_format))
                push!(values, parse(Float32, value_str))
            end
        end
        values = reshape(values, 1, length(values)) # single location
    end
    timeseries=TimeSeries(values, times, [name], [longitude], [latitude], quantity, source)
    return timeseries
end

function write_single_noos_file(filename::String, series::TimeSeries,index=1)
    # check index
    if (index < 1) || (index > length(get_names(series)))
        error("Index out of bounds: $index")
    end
    # check if file exists
    if isfile(filename)
        error("File already exists: $filename. Will not overwrite.")
    end
    # Open the file for writing
    open(filename, "w") do file
        # Write header
        println(file, "#------------------------------------------------------")
        println(file, "# Timeseries retrieved from the series_ml package")
        println(file, "# Created at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(file, "#------------------------------------------------------")
        println(file, "# Location    : $(get_names(series)[index])")
        println(file, "# Position    : ($(get_longitudes(series)[index]),$(get_latitudes(series)[index]))")
        println(file, "# Source      : $(get_source(series))")
        println(file, "# Unit        : $(get_quantity(series))")
        println(file, "# Timezone    : GMT")
        println(file, "#------------------------------------------------------")
        # Write data
        times = get_times(series)
        values = get_values(series)
        for i in 1:length(times)
            time_str = Dates.format(times[i], "yyyymmddHHMM")
            value_str = @sprintf("%.5f", values[index,i])
            println(file, "$time_str    $value_str")
        end
    end
end