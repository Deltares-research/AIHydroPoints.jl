# netcdf_utils.jl

using NetCDF
using Dates

const NAME_LEN_MAX = 256

"""
function waterlevel_series_to_netcdf(filename, times, waterlevels, station_names,station_x,station_y)

Write data to a netcdf file with the following variables and attributes (mimicing the Delft3D-FM history output format):
global:
    - title: "Water level time series"
    - institution: "Deltares"
    - source: "AI-Hydro"
    - history: "Created by Julia"
    - references: "
    - date_created: current date
    - conventions: "CF-1.5"
time (time):
    - standard_name: "time"
    - long_name: "time"
    - units: "seconds since 2000-01-01 00:00:00"
station_name (name_len,stations):
    - long_name: "station name"
    - cf_role: "timeseries_id"
station_x_coordinate (stations):
    - units: ""degrees_east"
    - long_name: "station x coordinate"
    - standard_name: "longitude"
station_y_coordinate (stations):
    - units: ""degrees_north"
    - long_name: "station y coordinate"
    - standard_name: "latitude"
waterlevel(stations,time):
    - standard_name: "sea_surface_height"
    - long_name: "water level"
    - units: "m"
    - coordinates: "station_x_coordinate station_y_coordinate station_name"
    - _FillValue: -999.0
    - missing_value: NaN
"""
function waterlevel_series_to_netcdf(filename, times, waterlevels, station_names,station_x=nothing,station_y=nothing)
    # check for station coordinates
    if isnothing(station_x)
        station_x=zeros(Float32,length(station_names))
    end
    if isnothing(station_y)
        station_y=zeros(Float32,length(station_names))
    end
    # check for file
    if isfile(filename)
         println("File $(filename) already exists, will not overwrite")
         return
    end 
    # global attributes
    gatts = Dict("title"=>"Water level time series",
                 "institution"=>"Deltares",
                 "source"=>"AI-Hydro",
                 "history"=>"Created by Julia",
                 "date_created"=>"$(Dates.now())",
                 "conventions"=>"CF-1.5")
    # create time dimansion
    times_secs_since = [t.value for t in times.-DateTime(2000,1,1,0,0,0) ]/1000.0#robust_timedelta_sec(times,DateTime(2000,1,1))
    time_atts = Dict("standard_name"=>"time","long_name"=>"time","units"=>"seconds since 2000-01-01 00:00:00")
    time_dim = NcDim("time",times_secs_since,time_atts)
    # create station dimension
    station_dim = NcDim("stations",length(station_names))
    # create name_len dimansion
    name_len_dim = NcDim("name_len",NAME_LEN_MAX)
    # create longitude variable
    station_x_atts = Dict("units"=>"degrees_east","long_name"=>"station x coordinate","standard_name"=>"longitude")
    station_x_var = NcVar("station_x_coordinate",[station_dim],atts=station_x_atts,t=Float32)
    # create latitude variable
    station_y_atts = Dict("units"=>"degrees_north","long_name"=>"station y coordinate","standard_name"=>"latitude")
    station_y_var = NcVar("station_y_coordinate",[station_dim],atts=station_y_atts,t=Float32)
    # create station name variable
    name_atts = Dict("long_name"=>"station name","cf_role"=>"timeseries_id")
    name_var = NcVar("station_name",[name_len_dim,station_dim],atts=name_atts,t=NC_CHAR)
    # create waterlevel variable
    waterlevel_atts = Dict("standard_name"=>"sea_surface_height",
                           "long_name"=>"water level",
                           "units"=>"m",
                           "coordinates"=>"station_x_coordinate station_y_coordinate station_name",
                           "_FillValue"=>-999.0f0,
                           "missing_value"=>NaN)
    waterlevel_var= NcVar("waterlevel",[station_dim,time_dim],atts=waterlevel_atts, t=Float32) #t=Float32

    # create netcdf file and write variables
    NetCDF.create(filename, NcVar[waterlevel_var,station_x_var,station_y_var,name_var],gatts=gatts,mode=NC_NETCDF4) do nc
        NetCDF.putvar(nc, "waterlevel", waterlevels)
        NetCDF.putvar(nc, "station_x_coordinate", station_x)
        NetCDF.putvar(nc, "station_y_coordinate", station_y)
        NetCDF.putvar(nc, "station_name", nc_string2char(station_names))
    end
end