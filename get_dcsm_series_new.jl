

cd(@__DIR__)

# activate the environment
using Pkg
Pkg.activate(".")
# Load required packages
using Dates
using AWS
using Minio
using URIs
using Rasters, ZarrDatasets

    url_or_filename = "s3://s3.deltares.nl/emodnet/DCSM-FM_0_5nm_1980-2023_his.zarr?profile=minio_deltares"
#    series = ZarrTimeSeries(urlname, "waterlevel")

# as in get_dcsm_series.jl
# base_url="s3://emodnet"
# dataset_url="$(base_url)/DCSM-FM_0_5nm_1980-2023_his.zarr"
# profile="minio_deltares" #switch to a specific profile instead of default
# c = AWS.AWSConfig(profile=profile)
# #mc = Minio.MinioConfig("https://s3.deltares.nl", c.credentials; region="eu-west-1") # use Minio.jl to point to correct server
# mc = Minio.MinioConfig("https://s3.deltares.nl", c.credentials, region=c.region) # use Minio.jl to point to correct server
# AWS.global_aws_config(mc) # set the global config to the minio server, because you can't pass the config to the Raster constructor yet

# Connect to the dataset
his_data=RasterStack(dataset_url;lazy=true)


    uri_or_path = URI(url_or_filename)
    zarr_url=nothing

    # Handle S3 URL
    # if !has_aws_credentials()
    #     error("AWS credentials not found. Please set up your AWS credentials in ~/.aws/credentials and ~/.aws/config.")
    # end
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

    zarr_url=String(zarr_url) #convert to String if it is not already

    println("Opening Zarr file or URL: $(zarr_url) / $(url_or_filename)")
    zarr_data = nothing
    try
        zarr_data = RasterStack(zarr_url; lazy=true)

#
# NOTE: This is just a part of a script. The rest is still to be implemented!
#