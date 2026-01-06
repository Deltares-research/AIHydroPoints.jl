# get_dcsm_series.jl
#
# Download a subset of a long DCSM simulation (1980-2023) from the  bucket on the Deltares Minio server.
# The data is stored in a Zarr dataset, which is a compressed, chunked, and parallelized format that is optimized for cloud storage.
# and save it to a local directory.

# Move to this folder if not already there
cd(@__DIR__)

# activate the environment
using Pkg
Pkg.activate(".")
# Load required packages
using Dates
using AWS
using Minio
using Rasters, ZarrDatasets, NCDatasets
using Plots
using JLD2


#
# data selection
#
# ↓ stations,
# → Ti       Sampled{DateTime} [1979-12-22T00:00:00, …, 2024-01-31T00:00:00] ForwardOrdered Irregular Points
# ├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── layers ┤
# :station_name         eltype: Union{Missing, Zarr.MaxLengthStrings.MaxLengthString{256, UInt8}} dims: stations size: 317
# :station_x_coordinate eltype: Union{Missing, Float64} dims: stations size: 317
# :station_y_coordinate eltype: Union{Missing, Float64} dims: stations size: 317
# :waterlevel           eltype: Union{Missing, Float64} dims: stations, Ti size: 317×2319985

# set station_names or station_ids to select a subset of the data
station_names = ["VLISSGN","HOEKVHLD","DENHDR","DELFZL","HARLGN"] # or [] to use station_ids
station_ids = []   # collect(1:317) #[] # or [] to use station_names
# station_ids=collect(1:50)


# set start_date and end_date to select a subset of the data
time_skip=6 # reduce number of time steps to save memory needed op GPU
# start_date = DateTime(2008,1,1) #training
# end_date = DateTime(2011,1,1)
# start_date = DateTime(2011,1,1) #testing
# end_date = DateTime(2012,1,1)
start_date = DateTime(2012,1,1) #testing
end_date = DateTime(2013,1,1)

# save to this file
# output_file = "DCSM-FM_0_5nm_2008_3yr_5stations_his.jld2"
# output_file = "DCSM-FM_0_5nm_2011_5stations_his.jld2"
output_file = "DCSM-FM_0_5nm_2012_5stations_his.jld2"
# output_file = "DCSM-FM_0_5nm_2010_317stations_his.jld2"
# output_file = "DCSM-FM_0_5nm_2011_317stations_his.jld2"
# output_file = "DCSM-FM_0_5nm_2008_3yr_317stations_his.jld2"
# output_file = "DCSM-FM_0_5nm_2011_50stations_his.jld2"
# output_file = "DCSM-FM_0_5nm_2008_3yr_50stations_his.jld2"

#
# Note: this script will only work if you have the necessary credentials to access the Scaleway bucket.
# these should be in the .aws/credentials file in your home directory. It should contain a section like this:
# [minio_deltares]
# aws_access_key_id=3qM1e6veyLD0CrGHkgFq# 
# aws_secret_access_key = blablablablaetcetracetera
#
# and the .aws/config file should contain the following:
#
# [profile minio_deltares]
# region=eu-west-1
# output=json

# Set up the connection to the Scaleway bucket
base_url="s3://emodnet"
dataset_url="$(base_url)/DCSM-FM_0_5nm_1980-2023_his.zarr"
profile="minio_deltares" #switch to a specific profile instead of default
c = AWS.AWSConfig(profile=profile)
#mc = Minio.MinioConfig("https://s3.deltares.nl", c.credentials; region="eu-west-1") # use Minio.jl to point to correct server
mc = Minio.MinioConfig("https://s3.deltares.nl", c.credentials, region=c.region) # use Minio.jl to point to correct server
AWS.global_aws_config(mc) # set the global config to the minio server, because you can't pass the config to the Raster constructor yet

# Connect to the dataset
his_data=RasterStack(dataset_url;lazy=true)

# Select a subset of the data

function lookup_station_ids(station_names, all_station_names)
    selected_ids = Vector{Int}()
    selected_names = Vector{String}()
    for station_name in station_names
        station_id = findfirst(x->x==station_name, all_station_names)
        if isnothing(station_id)
            println("Station $station_name not found")
        else
            #@show station_id, station_name
            push!(selected_ids, station_id)
            push!(selected_names, station_name)
        end
    end
    return (selected_names,selected_ids)
end

function lookup_station_names(station_ids, all_station_names)
    selected_names = Vector{String}()
    selected_ids = Vector{Int}()
    for station_id in station_ids
        #@show station_id
        push!(selected_names, all_station_names[station_id])
        push!(selected_ids, station_id)
    end
    return (selected_names,selected_ids)
end

# Select a subset of the data
his_data_timesel = his_data[Ti=start_date..end_date]

# get the station names
all_station_names = his_data_timesel.station_name[:]

if length(station_names) > 0
    selected_names,selected_ids = lookup_station_ids(station_names, all_station_names)
else
    if length(station_ids) == 0
        station_ids = collect(1:length(all_station_names))
    end
    selected_names,selected_ids = lookup_station_names(station_ids, all_station_names)
end

waterlevel_temp = his_data_timesel.waterlevel
ntimes = size(waterlevel_temp,2)
nstations = selected_names
waterlevel=zeros(Float32,length(selected_ids),ntimes)
for i in 1:length(selected_ids)
    waterlevel[i,:] = waterlevel_temp[selected_ids[i],:]
end

station_names = selected_names
station_x_coordinate = his_data_timesel.station_x_coordinate.data
station_x_coordinate = station_x_coordinate[selected_ids]
station_y_coordinate = his_data_timesel.station_y_coordinate.data
station_y_coordinate = station_y_coordinate[selected_ids]
times = collect(dims(his_data_timesel,Ti)).data
if time_skip > 1
    times = times[1:time_skip:end]
    waterlevel = waterlevel[:,1:time_skip:end]
end

# Save the data
save(output_file,Dict("waterlevel"=>waterlevel,"station_names"=>station_names,"station_x_coordinate"=>station_x_coordinate,"station_y_coordinate"=>station_y_coordinate,"times"=>times))

# Load the data for testing
# data = load(output_file)
# @show keys(data)