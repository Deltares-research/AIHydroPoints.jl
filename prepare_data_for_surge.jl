# prepare_data_for_surge.jl
#
# The surge component of the ML model is driven by winds and air pressure.
# This scipts prepares data for training, validation, and testing.


# switch to this folder as the working directory if not already here
cd(@__DIR__)
# activate the environment
using Pkg
Pkg.activate(".")

# packages
using NCDatasets
using Dates
using Plots
using JLD2
include("wind_stress.jl")

# parameters
data_folder = "./era5_north_sea_2008_2012"
labels=["training","testing","validation"]
tstart_values=[DateTime(2008,1,1),DateTime(2011,1,1),DateTime(2012,1,1)]
tend_values=[DateTime(2011,1,1),DateTime(2012,1,1),DateTime(2012,12,31,23,00)]
# points for input data
x_points = [ 3.0, 3.75, 4.25, 5.25, 6.5, 0.0,  5.0, 0.0, 0.0]
y_points = [51.5,52.0 ,53.0 ,53.25,53.75,56.0,56.0,60.0,50.25]


# functions

function nearest_index(x_point::Number,x)
    return argmin(abs.(x .- x_point))
end
function nearest_index(x_points::Vector,x)
    indices=[nearest_index(x_point,x) for x_point in x_points]
    return indices
end

# Open dataset with wind_stress
files=readdir(data_folder)
files=filter(x->occursin(r"era5_wind_.*.nc",x),files)
file_paths=joinpath.(data_folder,files)
d=NCDataset(file_paths;aggdim="valid_time")
era5_times=d["valid_time"][:]
era5_longitude=d["longitude"][:]
era5_latitude=d["latitude"][:]
# create subsets for training, testing, and validation 
for idataset=1:length(labels)
    label=labels[idataset]
    println("Processing $(label)")
    tstart=tstart_values[idataset]
    tend=tend_values[idataset]
    # indices for time
    ifirst = findfirst(x->x>=tstart,era5_times)
    ilast = findlast(x->x<=tend,era5_times) 
    #@show ifirst,ilast
    #@show era5_times
    times=era5_times[ifirst:ilast]
    # indices for points
    m_points=nearest_index(x_points,era5_longitude)
    n_points=nearest_index(y_points,era5_latitude)
    # create subsets
    wind_x = zeros(length(m_points),ilast-ifirst+1)
    wind_y = zeros(length(m_points),ilast-ifirst+1)
    pressure = zeros(length(m_points),ilast-ifirst+1)
    for i in 1:length(m_points)
        wind_x[i,:] = d["u10"][m_points[i],n_points[i],ifirst:ilast]
        wind_y[i,:] = d["v10"][m_points[i],n_points[i],ifirst:ilast]
        pressure[i,:] = d["msl"][m_points[i],n_points[i],ifirst:ilast]
    end
    # convert to stress
    stress_x = similar(wind_x)
    stress_y = similar(wind_y)
    for index in eachindex(wind_x)
        w_x=wind_x[index]
        w_y=wind_y[index]
        s_x,s_y = uv_to_stress_xy(w_x,w_y)
        stress_x[index] = s_x
        stress_y[index] = s_y
    end
    # save to file
    output_file = "era5_wind_stress_$(year(tstart))_$(label).jld2"
    save(output_file,Dict("pressure"=>pressure,"stress_x"=>stress_x,"stress_y"=>stress_y,"station_x_coordinate"=>x_points,"station_y_coordinate"=>y_points,"times"=>times))
end
