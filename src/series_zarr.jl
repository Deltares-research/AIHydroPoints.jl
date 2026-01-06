# series_zarr.jl

using Dates
using Rasters, ZarrDatasets, NCDatasets
using Minio, AWS
using URIs


# Define values structure for NetCDF time series
# use lazy loading for the values
struct ZarrTimeSeries <: AbstractTimeSeries
    dataset::RasterStack
    filename::String
    quantity::String # type of data, e.g. "waterlevel"
    source::String # source of the data, e.g. "Zarr file: <filename>
    stations::Vector{String} # names of the stations
    times::Vector{DateTime} # time steps
end

# check for presence of .aws credentials and config
function has_aws_credentials()
    filename_credentials=AWS.dot_aws_credentials_file()
    filename_config=AWS.dot_aws_config_file()
    if !isfile(filename_credentials) || !isfile(filename_config)
        println("No AWS credentials or config file found. Please set up your AWS credentials.")
        println("Your AWS credentials file should be located at: $(filename_credentials) and contain something like:")
        println("[minio_deltares]")
        println("aws_access_key_id=3q23e6rgyLD0CrRGkgFq")
        println("aws_secret_access_key=blablablablaetcetracetera")
        println("Your AWS config file should be located at: $(filename_config) and contain something like:")
        println("[profile minio_deltares]")
        println("region=eu-west-1")
        println("output=json")
        return false
    end
    return isfile(joinpath(homedir(), ".aws", "credentials")) && isfile(joinpath(homedir(), ".aws", "config"))
end

# constructor for ZarrTimeSeries from filename and quantity
"""
function ZarrTimeSeries(url_or_filename::String, quantity::String, source::String="")
Creates a ZarrTimeSeries object from a Zarr file and a specified quantity.
use: ts = ZarrTimeSeries("path/to/file.zarr", "waterlevel")
where url_or_filename is the path to the Zarr file and quantity is the variable name in the file.
Returns a ZarrTimeSeries object that can be used with the series_ml interface.
The default source is "Zarr file: <url_or_filename>".

The url_or_filename can also be a URL to a Zarr file, for example on a web server or S3 bucket.
Example URLs:
- "https://example.com/path/to/file.zarr"
- "s3://minio.example.com/bucket-name/path/to/file.zarr?profile=minio_example_com"
- "local_folder/file.zarr"
If the URL is an S3 URL, it will use the AWS credentials from the ~/.aws/credentials and ~/.aws/config files.
"""
function ZarrTimeSeries(url_or_filename::String, quantity::String, source::String="")
    uri_or_path = URI(url_or_filename)
    zarr_url=nothing
    if uri_or_path.scheme == "http" || uri_or_path.scheme == "https"
        # Handle HTTP URL
        zarr_url = url_or_filename
    elseif uri_or_path.scheme == "s3"
        # Handle S3 URL
        if !has_aws_credentials()
            error("AWS credentials not found. Please set up your AWS credentials in ~/.aws/credentials and ~/.aws/config.")
        end
        params=queryparams(uri_or_path)
        aws_profile = get(params, "profile", "default") # use default profile if not
        println("Using AWS profile: $(aws_profile) for S3 access.")
        c = AWS.AWSConfig(profile=aws_profile) # read the default AWS credentials from the environment or config file at .aws/config
        server_url = "https://$(uri_or_path.host)"
        println("Connecting to S3 server: $(server_url)")
        mc = Minio.MinioConfig(server_url, c.credentials; region=c.region)
        AWS.global_aws_config(mc) # set the global config to the minio server
        zarr_url = "$(uri_or_path.scheme):/$(uri_or_path.path)"
        println("Zarr URL: $(zarr_url)")
    elseif uri_or_path.scheme == "" # Local file path
        zarr_url = uri_or_path.path
        if !isdir(zarr_url)
            error("Zarr folder $(zarr_url) does not exist.")
        end
    end
    zarr_url=String(zarr_url) #convert to String if it is not already

    println("Opening Zarr file or URL: $(zarr_url) / $(url_or_filename)")
    zarr_data = nothing
    try
        zarr_data = RasterStack(zarr_url; lazy=true)
    catch e
        error("Failed to open Zarr file or url at $(zarr_url) which originated from $(url_or_filename): $(e)")
    end
    if length(source)==0
        source = "Zarr file: $(zarr_url)"
    end
    # cache some metadata
    println("Reading metadata from Zarr file...")
    stations_src= zarr_data["station_name"][:]
    stations = fill("", length(stations_src)) # preallocate
    stations .= stations_src # copy station names to String array; avoid for loop
    times_src= dims(zarr_data, Ti)[:]
    times = fill(DateTime(1900), length(times_src)) # preallocate
    times .= times_src # copy times to DateTime array; avoid for loop
    println("Caching done.")
    return ZarrTimeSeries(zarr_data, url_or_filename, quantity, source, stations, times)
end

#
# getters for the fields
#
function get_values(ts::ZarrTimeSeries)
    # Read the values now for all stations and times
    values=ts.dataset[ts.quantity][:,:]
    result = Array{Float32}(undef, size(values))
    @. result .= values #copy values to Float32 array
    return result
end

function get_times(ts::ZarrTimeSeries)
    # Read the times now for all stations
    times = ts.times
    return times
end

function get_names(ts::ZarrTimeSeries)
    # Read the names now for all stations
    # names = ts.dataset["station_name"][:]
    # nstations = length(names)
    # if nstations == 0
    #     return String[]
    # end
    # result= Vector{String}(undef, nstations)
    # result .= names # copy names to String array; avoid for loop
    result = ts.stations
    return result
end

function get_longitudes(ts::ZarrTimeSeries)
    # Read the longitudes now for all stations
    if length(size(ts.dataset["station_x_coordinate"])) == 2
        longitudes = ts.dataset["station_x_coordinate"][:,1] # ignore time dependency
    else
        longitudes = ts.dataset["station_x_coordinate"][:] # Handle case where longitudes are not time-dependent
    end
    nstations = length(longitudes)
    if nstations == 0
        return Float64[]
    end
    result= Vector{Float64}(undef, nstations)
    result .= longitudes # copy longitudes to Float64 array; avoid for loop
    return result
end

function get_latitudes(ts::ZarrTimeSeries)
    # Read the latitudes now for all stations
    if length(size(ts.dataset["station_y_coordinate"])) == 2
        latitudes = ts.dataset["station_y_coordinate"][:,1] #ingore time dependency
    else
        latitudes = ts.dataset["station_y_coordinate"][:] # Handle case where latitudes are
    end
    nstations = length(latitudes)
    if nstations == 0
        return Float64[]
    end
    result= Vector{Float64}(undef, nstations)
    result .= latitudes # copy latitudes to Float64 array; avoid for loop
    return result
end

function get_quantity(ts::ZarrTimeSeries)
    return ts.quantity
end

function get_source(ts::ZarrTimeSeries)
    return ts.source
end

#
# Selection methods
# These are the implemtentations for the ZarrTimeSeries, that override the default implementations in series.jl.
# In this implementation, we read the values only for the selected locations and times, and store the result in memory.
#
# function select_locations_by_ids(ts::ZarrTimeSeries, location_indices::Vector{T} where T<:Integer)
#     selected_names = get_names(ts)[location_indices]
#     selected_longitudes = get_longitudes(ts)[location_indices]
#     selected_latitudes = get_latitudes(ts)[location_indices]
#     selected_times = get_times(ts)
#     selected_quantity = get_quantity(ts)
#     selected_source = get_source(ts)
#     #selected_values = get_values(ts)[location_indices, :] # This would read all values before selection
#     zarr_values = ts.dataset[ts.quantity]
#     n_stations,n_times = size(zarr_values)
#     selected_values = zeros(Float32, length(location_indices), n_times) # Preallocate for selected values
#     # copy station by station
#     for (i, loc) in enumerate(location_indices) # this will read only the values for the selected locations
#         selected_values[i, :] .= zarr_values[loc, :]
#     end
#     # Create a new TimeSeries object with the selected values
#     return TimeSeries(selected_values, selected_times, selected_names, selected_longitudes, selected_latitudes,
#                       selected_quantity, selected_source)
# end

function select_locations_by_ids(ts::ZarrTimeSeries, location_indices::Vector{T} where T<:Integer)
    selection=ts.dataset[stations=location_indices] #Lazy selection of locations
    return ZarrTimeSeries(selection, ts.filename, ts.quantity, ts.source, ts.stations[location_indices], ts.times)
end

function select_timespan(ts::ZarrTimeSeries, start_time::DateTime, end_time::DateTime)
    selection=ts.dataset[Ti=start_time..end_time] # lazy selection of times
    times_src= dims(selection, Ti)[:]
    times = fill(DateTime(1900), length(times_src)) # preallocate
    times .= times_src # copy times to DateTime array; avoid for loop
    return ZarrTimeSeries(selection, ts.filename, ts.quantity, ts.source, ts.stations, times)
end

function select_times_by_ids(ts::ZarrTimeSeries, time_indices::Vector{T} where T<:Integer)
    if isempty(time_indices)
        error("No time indices provided.")
    end
    selection = ts.dataset[Ti=time_indices] # Lazy selection of times
    times_src= dims(selection, Ti)[:]
    times = fill(DateTime(1900), length(times_src)) # preallocate
    times .= times_src # copy times to DateTime array; avoid for loop
    return ZarrTimeSeries(selection, ts.filename, ts.quantity, ts.source, ts.stations, times)
end


#
# Show function for ZarrTimeSeries
#
function Base.show(io::IO, series::ZarrTimeSeries)
    println(io, "ZarrTimeSeries:")
    println(io, "   Filename: ", series.filename)
    println(io, "   Quantity: ", series.quantity)
    println(io, "   Source: ", series.source)
    println(io, "   Number of locations: ", length(series.stations))
    println(io, "   Number of time points: ", length(series.times))
    println(io, "   Data shape: ", size(series.dataset[series.quantity]))
    println(io, "   Times: ", series.times[1], " to ", series.times[end])
    println(io, "   Locations: ", join(series.stations, ", "))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", series::ZarrTimeSeries)
    println(io, "ZarrTimeSeries: $(series.quantity) from $(series.source), with $(length(series.stations)) locations, from $(series.times[1]) until $(series.times[end]).")
    #show(io, series)
    return nothing
end

function finalize(ts::ZarrTimeSeries)
    # Clean up resources if needed
    finalize(ts.dataset) # Finalize the dataset to release resources
    return nothing
end