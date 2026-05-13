# get_era_series.jl
# Example script to get ERA5 time series from Deltares Minio S3 server. The data is stored in a Zarr dataset, which is a compressed, chunked format that is optimized for cloud storage. The script downloads a selection and saves it to a local file.
# To run this script, you need to have access to the Deltares Minio server with proper AWS credentials. AWS credentials should be set up in ~/.aws/credentials and ~/.aws/config.

cd(@__DIR__)

# activate the environment
using Pkg
Pkg.activate(".")
# Load required packages
using Dates
using AIHydroPoints
using AWS
using Minio
using Rasters, ZarrDatasets
using ProgressMeter

# The url_or_filename can also be a URL to a Zarr file, for example on a web server or S3 bucket.
# Example URLs:
# - "https://example.com/path/to/file.zarr"
# - "s3://minio.example.com/bucket-name/path/to/file.zarr?profile=minio_example_com"
# - "local_folder/file.zarr"
# The profile parameter in the URL is used to specify which AWS credentials profile to use when accessing the S3 bucket. This allows you to manage multiple sets of credentials for different S3 buckets or services.
hostname = "s3.deltares.nl"
aws_profile="minio_deltares"
bucket = "ai-hydro"
path = "dcsm_1980_2023/era5_north_sea_1980_2023.zarr"
quantities=["10m_u_component_of_wind","10m_v_component_of_wind","mean_sea_level_pressure"]
source="ERA5_North_Sea_1980-2023"

# open the Zarr time series on S3
if !has_aws_credentials()
    error("AWS credentials not found. Please set up your AWS credentials in ~/.aws/credentials and ~/.aws/config.")
end
println("Using AWS profile: $(aws_profile) for S3 access.")
c = AWS.AWSConfig(profile=aws_profile) # read the default AWS credentials from the environment or config file at .aws/config
server_url = "https://$(hostname)"
println("Connecting to S3 server: $(server_url)") # should look something like "https://s3.deltares.nl"
mc = Minio.MinioConfig(server_url, c.credentials; region=c.region)
AWS.global_aws_config(mc) # set the global config to the minio server
zarr_url = "s3://$(bucket)/$(path)"
println("Zarr URL: $(zarr_url)") # should look something like "s3://ai-hydro/era5_north_sea_1980_2023/era5_north_sea_1980_2023.zarr"
zarr_data = RasterStack(zarr_url; lazy=true)
# RasterStack can also work with local Zarr files, for example:
# zarr_data = RasterStack("local_folder/file.zarr"; lazy=true)

#
# Selection parameters
#
# time range
tstart=DateTime(2010,1,1)
tend=DateTime(2011,1,1)
# points coordinates
x_points = [ 3.0, 3.75, 4.25, 5.25, 6.5, 0.0,  5.0, 0.0, 0.0]
y_points = [51.5,52.0 ,53.0 ,53.25,53.75,56.0,56.0,60.0,50.25]
# filenames for output
output_files = ["era5_2010_9points_$(quantity).jld2" for quantity in ["wind_stress_x","wind_stress_y","mean_sea_level_pressure"]]

function download_points_from_maps(dataset,variable_name,start_time::DateTime,end_time::DateTime,x_points,y_points,source,time_chunksize=240)
    println("Downloading variable $(variable_name) for points and time range...")
    # if !(Symbol(variable_name) in keys(dataset))
    #     error("Variable $(variable_name) not found in the dataset.")
    # end
    variable = dataset[variable_name]
    # get time selection
    alltimes = collect(lookup(variable,Ti))
    time_indices = findall(t -> t >= start_time && t <= end_time, alltimes)
    if isempty(time_indices)
        error("No time steps found in the specified timespan.")
    end
    itime_first = time_indices[1]
    itime_last = time_indices[end]
    ntimes=length(time_indices)
    times=alltimes[itime_first:itime_last]
    # create empty array for the selected data
    values = zeros(length(x_points),ntimes)
    # get coordinates
    x_coords = dims(variable, X)[:]
    y_coords = dims(variable, Y)[:]
    # loop over time chunks
    if time_chunksize*size(variable,1)*size(variable,2) > 1e8
        time_chunksize = Int(floor(1e8/(size(variable,1)*size(variable,2))))
        println("Time chunk size too large, reducing to $(time_chunksize) to avoid memory issues.")
    end
    # download data in chunks to avoid memory issues
    @showprogress for itime in itime_first:time_chunksize:itime_last
        itime_end = min(itime+time_chunksize-1,itime_last)
        buffer=variable[:,:,itime:itime_end] # load a chunk of data into memory
        for ipoint in 1:length(x_points)
            x_point = x_points[ipoint]
            y_point = y_points[ipoint]
            # find nearest grid point
            point_in_buffer=buffer[X(Near(x_point)),Y(Near(y_point))]
            # copy data for the selected time steps and location to the local array
            values[ipoint,(itime-itime_first+1):(itime_end-itime_first+1)] = point_in_buffer[:]
        end
    end
    # create names based on point coordinates
    names = ["$(variable_name)_$(x_points[i])_$(y_points[i])" for i in 1:length(x_points)]
    # create metadata for the time series
    longitues = x_points
    latitudes = y_points
    quantity = variable_name
    # Create a time series object with the selected data and metadata
    time_series = TimeSeries(values, times, names, longitues, latitudes, quantity, source)
    return time_series
end

# download the selected data for the specified points and time range
u10_series = download_points_from_maps(zarr_data, "10m_u_component_of_wind", tstart, tend, x_points, y_points, source)
v10_series = download_points_from_maps(zarr_data, "10m_v_component_of_wind", tstart, tend, x_points, y_points, source)
msl_series = download_points_from_maps(zarr_data, "mean_sea_level_pressure", tstart, tend, x_points, y_points, source)

# convert wind components to stress
wind_x = get_values(u10_series)
wind_y = get_values(v10_series)
stress = uv_to_stress_xy.(wind_x, wind_y) # apply the conversion to each element of the arrays
stress_x=first.(stress) # extract the x component of the stress
stress_y=last.(stress) # extract the y component of the stress
# create new time series for the stress
stress_x_series = TimeSeries(stress_x, get_times(u10_series), get_names(u10_series), get_longitudes(u10_series), get_latitudes(u10_series), "wind_stress_x", source)
stress_y_series = TimeSeries(stress_y, get_times(v10_series), get_names(v10_series), get_longitudes(v10_series), get_latitudes(v10_series), "wind_stress_y", source)

# Save to local file
write_to_jld2(stress_x_series, output_files[1])
write_to_jld2(stress_y_series, output_files[2])
write_to_jld2(msl_series, output_files[3])
println("Data saved to local files: $(output_files).")