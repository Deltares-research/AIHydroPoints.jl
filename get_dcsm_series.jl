# get_dcsm_series_new.jl
# Example script to get DCSM time series from Deltares Minio S3 server.
# The data is stored in a Zarr dataset, which is a compressed, chunkedformat that is optimized for cloud storage.
# The script downloads a selection and saves it to a local file.
# To run this script, you need to have access to the Deltares Minio server with proper AWS credentials.
# AWS credentials should be set up in ~/.aws/credentials and ~/.aws/config.

cd(@__DIR__)

# activate the environment
using Pkg
Pkg.activate(".")
# Load required packages
using Dates
using AIHydroPoints

# The url_or_filename can also be a URL to a Zarr file, for example on a web server or S3 bucket.
# Example URLs:
# - "https://example.com/path/to/file.zarr"
# - "s3://minio.example.com/bucket-name/path/to/file.zarr?profile=minio_example_com"
# - "local_folder/file.zarr"
url_or_filename = "s3://s3.deltares.nl/ai-hydro/dcsm_1980_2023/DCSM-FM_0_5nm_1980-2023_his.zarr?profile=minio_deltares"

# open the Zarr time series
quantity="waterlevel"
source="DCSM-FM_0_5nm_1980-2023"
his_data=ZarrTimeSeries(url_or_filename, quantity, source)

# Select time range
start_date = DateTime(2012,1,1)
end_date = DateTime(2013,1,1)
println("Selecting time range from $(start_date) to $(end_date) and downloading data. Please wait...")
his_data_timesel = select_timespan(his_data, start_date, end_date)
println("Time range selected.")
# Select locations by names or ids
station_names = ["VLISSGN","HOEKVHLD","DENHDR","HARLGN","DELFZL"] # or [] to use station_ids
all_station_names = get_names(his_data_timesel)
station_ids = find_location_index(station_names, all_station_names)
his_data_selected = select_locations_by_ids(his_data_timesel, station_ids)


# Save to local file
output_file = "DCSM-FM_0_5nm_2012_5stations_his.jld2"
if isfile(output_file)
    rm(output_file)
    # error("Output file $(output_file) already exists. Please remove it before running this script.")
end
write_to_jld2(his_data_selected, output_file)
if isfile(output_file)
    println("Output file $(output_file) written successfully.")
else
    error("Failed to write output file $(output_file).")
end

# Optionally, check if it was written correctly
# local_series = JLD2TimeSeries(output_file)
# @assert get_values(local_series) == get_values(his_data_selected)
# @assert get_times(local_series) == get_times(his_data_selected)
# @assert get_names(local_series) == get_names(his_data_selected)
# @assert get_longitudes(local_series) == get_longitudes(his_data_selected)
# @assert get_latitudes(local_series) == get_latitudes(his_data_selected)
# @assert get_quantity(local_series) == get_quantity(his_data_selected)
# @assert get_source(local_series) == get_source(his_data_selected)

println("DCSM time series saved to $(output_file).")