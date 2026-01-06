#learn_waves.jl
#
# Train a model to predict waves from winds/wind-stress

# Move to this folder if not already there
cd(@__DIR__)
# activate the environment
using Pkg
Pkg.activate(".")

# Force the use of the CPU
force_cpu=false #false

# Load required packages
using series_ml
using Dates
using Plots
using JLD2
using Statistics
using LinearAlgebra
using Flux
using CUDA, cuDNN
using ProgressMeter
using BSON
using NetCDF
using DataFrames
using CSV
# include("netcdf_utils.jl")
include("wind_stress.jl") #for wind to stress conversion

# Use the GPU or not
if CUDA.functional()
    @info "CUDA functional"
    println("CUDA version: ", CUDA.versioninfo())
    #println("Device: ", CUDA.devices())
    CUDA.allowscalar(false)
    device = gpu
else
    @info "Using CPU"
    device = cpu
end
if force_cpu==true #use CPU for debugging (even if CUDA is available)
    device=cpu
end

#
# settings for the input data
#
# data_folder=joinpath(@__DIR__,"data","waves_2021_2024","2021") #test just 2021
# data_folder=joinpath(@__DIR__,"data","waves_2021_2024")
# data_folder=joinpath(@__DIR__,"data","waves_2021_2024_7to3")
data_folder=joinpath(@__DIR__,"data","waves_2021_2024_10to11")
series_collection=NoosTimeSeriesCollection(data_folder)
@show get_sources(series_collection)
@show get_quantities(series_collection)
u10=get_series_from_collection(series_collection,"knmi_harmonie40_wind","wind_speed")
udir=get_series_from_collection(series_collection,"knmi_harmonie40_wind","wind_direction")
swh=get_series_from_collection(series_collection,"swan_dcsm_harmonie","wave_height")
# select a few locations and order them
output_locations=get_names(swh)
@show output_locations
input_locations=get_names(u10)
@show input_locations
u10=select_locations_by_names(u10,input_locations)
udir=select_locations_by_names(udir,input_locations)
swh=select_locations_by_names(swh,output_locations)
# select a common time range and fill missing values with 0.0f0
time_selection=DateTime(2021,1,1):Hour(1):DateTime(2024,11,1,0) # common time range with valid data
u10=select_timerange_with_fill(u10,time_selection,fill_value=0.0f0)
udir=select_timerange_with_fill(udir,time_selection,fill_value=0.0f0)
swh=select_timerange_with_fill(swh,time_selection,fill_value=0.0f0)



#
# ml model settings
#
# nchannel=(32,32,32,1)
nchannel=(64,64,64,1)

n_input_channels=64 #number of channels in the first layer of the model
f_activation=swish #swish or relu
nlayers=length(nchannel) #number of layers
const nlags=16 #time-steps used including the current time step
@assert nlags == 2^length(nchannel) "nlags must be a power of 2, specifically 2^length(nchannel)"
# const wind_scale=10.0f0 #scale wind to be of order 1
const wind_scale=0.5f0 #scale if set to stress
const wave_scale=3.0f0 #scale wave height to be of order 1
# some meta data
n_output_stations=length(output_locations)
n_input_vars=2 # wind_x, wind_y
n_input_stations=length(input_locations)
npars=n_input_vars*n_input_stations # data values per time lag

#
# optimizer settings
#
const nepochs=100 #50
const nbatch=256 
const learning_rate = 0.001 # initial learning_rate
const learning_rate_decay = 0.03 # lower learning rate at last eopch with factor
const learning_rate_steps=10 # apply every n steps
const regularization_weight=1.0f-4 #1.0e-4
const input_noise_std=0.30f0 #0.01f0 for wind, 0.001f0 for stress #add noise to the input data for regularization
const validation_fractions=0.25 #fraction of data used for validation
runid="10to11_explyr3"
const model_name="wave_model_$(runid)_3stations_$(nlayers)lyr_3yr_$(nbatch)batch_0p00005reg_$(nlags)lags_$(nepochs)epochs"

#
# prepare the data for training
#

"""
function times_to_timerange(times)

Convert a vector of DateTime to a DateTime range with regular time steps.
This assumes that the time steps are regular, but some time steps may be missing.
"""
function times_to_timerange(times)
    # estimate timestep
    #dt=median(diff(times)) #median doesn't work on Millisecond
    dt=median(Dates.value.(diff(times))) #in Milliseconds
    @info "Estimated time step: $(dt)"
    # create a complete range of times from the first to the last time with the estimated timestep
    return times[1]:Millisecond(dt):times[end]
end

# prepare the data for training
"""
function prepare_wave_data(data_windspeed,data_winddirection,data_waveheight,nlags)
Prepare the data for training the wave model.
The input data consists of wind speed and wind direction time series at multiple locations.
The output data consists of wave height time series at multiple locations.
The input to the model is a combination of:
- a one-hot encoding of the station index
- a time series of wind speed and wind direction at multiple locations for the past `nlags` time steps
The output of the model is the wave height at the target location at the current time step.
nlags: number of time steps to use as input to the model (including the current time step)
"""
function prepare_wave_data(data_windspeed,data_winddirection,data_waveheight,nlags)
    #time
    target_times = get_times(data_waveheight)
    target_timerange = times_to_timerange(target_times)
    tstart=target_timerange[1]
    tend=target_timerange[end]
    dt=step(target_timerange)
    wind_times = get_times(data_winddirection)
    wind_timerange = times_to_timerange(wind_times)
    # check that the timesteps, start and end times are the same
    @assert step(target_timerange) == step(wind_timerange) "Time steps of wind and wave data are not the same"
    @assert tstart == wind_timerange[1] "Start times of wind and wave data are not the same"
    @assert tend == wind_timerange[end] "End times of wind and wave data are not the same"
    itraining_times = nlags:length(target_timerange)
    ntraining_times = length(itraining_times)
    # locations
    target_locations = get_names(data_waveheight)
    source_locations = get_names(data_windspeed)
    n_target_locations = length(target_locations)
    n_source_locations = length(source_locations)
    # get data and scale
    swh_values = get_values(data_waveheight)

    u10_values = get_values(data_windspeed)
    udir_values = get_values(data_winddirection)
    n_input_vars=2 # wind_x, wind_y
    # convert wind to east and north components, wind direction is the direction from which the wind blows counting from north to east
    wind_x = u10_values .* -sind.(udir_values)
    wind_y = u10_values .* -cosd.(udir_values)
    # scale the wind speed to be of order 1
    wind_x = wind_x ./ wind_scale
    wind_y = wind_y ./ wind_scale
    # convert wind to stress components
    for i in eachindex(wind_x)
        wind_x[i], wind_y[i] = uv_to_stress_xy(wind_x[i], wind_y[i])
    end
    # scale the wave height to be of order 1
    swh_values = swh_values ./ wave_scale
    # create input data
    # The first input to the model is a one-hot encoding of the station index
    # It's a list of all combinations of stations and training times 
    station_index = collect(1:n_target_locations)*ones(ntraining_times)'
    x_station = Flux.onehotbatch(station_index[:], 1:n_target_locations)
    # Create list of all combinations of stations and training times
    all_times= [itime for i in 1:n_source_locations, itime in itraining_times][:]
    # create input data
    x_input=zeros(Float32,nlags,n_source_locations*n_input_vars,n_target_locations*ntraining_times)
    for itime in itraining_times
        wind_x_block=wind_x[:,itime-nlags+1:itime]
        wind_y_block=wind_y[:,itime-nlags+1:itime]
        x_block=Float32.(vcat(wind_x_block,wind_y_block))'
        for istation in 1:n_target_locations # we need a copy of the data for each station
            isample=((itime-nlags)*n_target_locations+istation)
            # @show itime,istation,isample
            # @show size(x_block), size(x_input[:,:,isample])
            x_input[:,:,isample].=x_block
        end
    end
    # create output data
    y_wave=reshape(swh_values[:,nlags:end],1,:) #flatten the waterlevel to a vector
    return (x_station, x_input, y_wave)
end

# prepare data for training
@info "Preparing data"
x_station,x_input,y = 
    prepare_wave_data(u10,udir,swh,nlags)

# remove any records with NaNs
function remove_nan_records(x_station, x_input, y)
    nrecords = size(y,2)
    valid_indices = [i for i in 1:nrecords if !any(isnan, x_input[:,:,i]) && !any(isnan, x_station[:,i]) && !isnan(y[1,i])]
    return x_station[:,valid_indices], x_input[:,:,valid_indices], y[:,valid_indices]
end
x_station, x_input, y = remove_nan_records(x_station, x_input, y)

# cut into training and validation sets
n_train=Int(floor((1.0-validation_fractions)*size(x_input,3)))
training_x_station_cpu = x_station[:,1:n_train]
training_x_input_cpu = x_input[:,:,1:n_train]
training_y_cpu = y[:,1:n_train]
validation_x_station_cpu = x_station[:,n_train+1:end]
validation_x_input_cpu = x_input[:,:,n_train+1:end]
validation_y_cpu = y[:,n_train+1:end]
@show size(training_x_station_cpu), size(training_x_input_cpu), size(training_y_cpu)
@show size(validation_x_station_cpu), size(validation_x_input_cpu), size(validation_y_cpu)

# move data to GPU if available
training_x_station=training_x_station_cpu |> device
training_x_input=training_x_input_cpu |> device
training_y=training_y_cpu |> device
validation_x_station=validation_x_station_cpu |> device
validation_x_input=validation_x_input_cpu |> device
validation_y=validation_y_cpu |> device

# Create a DataLoader
@info "Creating DataLoader"
dataloader = Flux.DataLoader((training_x_station,training_x_input,training_y), batchsize=nbatch, shuffle=true)


#
# Custom input layer
#
# struct WindInputLayer{T}
#     station_params::T
# end

# #constructor
# WindInputLayer(n_output_stations,nlags,npars) = WindInputLayer(
#     Dense(n_output_stations=>(nlags*npars),identity;bias=false)
# )

# # define the forward pass
# function (l::WindInputLayer)(x)
#     x_station, x_input = x
#     nlags,npars,nbatch= size(x_input)
#     s1 = l.station_params(x_station) 
#     # output is -(nlags*npars,nbatch)
#     # reshape to (nlags,npars,nbatch)
#     s1 = reshape(s1, (nlags, npars, nbatch))
#     z1 = s1 .* x_input
#     return z1
# end

struct WindInputLayer2{T1,T2}
    station_params::T1
    first_layer::T2
end

#constructor
WindInputLayer2(n_output_stations,nlags,npars,nchannels,f_activation) = WindInputLayer2(
    Dense(n_output_stations=>(nlags*nchannels),identity;bias=false),
    Conv((1,),npars=>nchannels,f_activation)
)

# define the forward pass
function (l::WindInputLayer2)(x)
    x_station, x_input = x 
    # Step 1: apply Conv layer to input
    x1= l.first_layer(x_input)
    # Step 2: Get station parameters for modulation
    s1 = l.station_params(x_station) 
    s1 = reshape(s1, size(x1)) # reshape the output of station_params to match x1
    # Step 3: Modulate the output of Conv layer with station parameters
    z1 = exp.(s1) .* x1
    return z1
end

#
# Define the wave model
#
# if length(nchannel)==1
#     wave_model = Chain(
#         WindInputLayer(n_input_stations,nlags,npars),
#         Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#         #Flux.flatten
#     )
# elseif length(nchannel)==2
#     wave_model = Chain(
#         WindInputLayer(n_input_stations,nlags,npars),
#         Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#         Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#         #Flux.flatten
#     )
# elseif length(nchannel)==3
#     wave_model = Chain(
#         WindInputLayer(n_input_stations,nlags,npars),
#         Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#         Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#         Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
#         #Flux.flatten
#     )
# elseif length(nchannel)==4
#     wave_model = Chain(
#         WindInputLayer(n_input_stations,nlags,npars),
#         Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#         Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#         Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
#         Conv((2,),nchannel[3]=>nchannel[4],relu,stride=(2,)),
#         Flux.flatten
#     )
# end

# model_cpu = Chain(
#     WindInputLayer(n_input_stations,nlags,npars),
#     Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#     Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#     Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
#     Conv((2,),nchannel[3]=>nchannel[4],identity,stride=(2,)),
#     Flux.flatten
# )

# model_cpu = Chain(
#     WindInputLayer(n_output_stations,nlags,npars),
#     Conv((2,),npars=>nchannel[1],f_activation,stride=(2,)),
#     Conv((2,),nchannel[1]=>nchannel[2],f_activation,stride=(2,)),
#     Conv((2,),nchannel[2]=>nchannel[3],f_activation,stride=(2,)),
#     Conv((2,),nchannel[3]=>nchannel[4],identity,stride=(2,)),
#     Flux.flatten
# )

#WindInputLayer2(n_output_stations,nlags,npars,nchannels,f_activation)
model_cpu = Chain(
    WindInputLayer2(n_output_stations,nlags,npars,n_input_channels,f_activation),
    Conv((2,),n_input_channels=>nchannel[1],f_activation,stride=(2,)),
    Conv((2,),nchannel[1]=>nchannel[2],f_activation,stride=(2,)),
    Conv((2,),nchannel[2]=>nchannel[3],f_activation,stride=(2,)),
    Conv((2,),nchannel[3]=>nchannel[4],identity,stride=(2,)),
    Flux.flatten
)

model=model_cpu |> device # move the model to the GPU if available, or keep it on the CPU

function compute_loss(model,x_station,x_input,y)
    y_hat = model((x_station, x_input))
    return Flux.mse(y_hat, y)
end

# ff=compute_loss(model_cpu,training_x_station_cpu,training_x_input_cpu,training_y_cpu)

# training loop 
@info "Training the model"
train_losses = []
val_losses = []
acc_losses = []
#opt_state = Flux.setup(Flux.Adam(learning_rate), model)
opt_state = Flux.setup(OptimiserChain(WeightDecay(regularization_weight), Adam(learning_rate)), model)

@showprogress for epoch in 1:nepochs
    @info "Epoch $(epoch)"
    loss=0.0f0
    for (x_station, x_input, y) in dataloader
        dloss, grads = Flux.withgradient(model) do m
            # add noise to the input data for regularization
            # TODO assumes x_input is on the CPU
            if input_noise_std > 0.0f0
                x_input_with_noise = x_input .+ input_noise_std * randn(Float32, size(x_input))
            end
            # Evaluate model and loss inside gradient context:
            y_hat = m((x_station, x_input_with_noise))
            Flux.mse(y_hat, y)
        end
        Flux.update!(opt_state, model, grads[1])
        loss += dloss
    end
    println("Epoch $epoch, accumulated loss: $(loss)")
    push!(acc_losses, loss)  # logging, outside gradient context
    @info "compute training loss"
    model_cpu=model |> cpu
    # train_loss=compute_loss(model,training_x_station_gpu,training_x_input_gpu,training_y_gpu)
    train_loss=compute_loss(model_cpu,training_x_station,training_x_input,training_y)
    push!(train_losses, train_loss)
    @info "compute validation loss"
    # val_loss=compute_loss(model,validation_x_station_gpu,validation_x_input_gpu,validation_y_gpu)
    val_loss=compute_loss(model_cpu,validation_x_station,validation_x_input,validation_y)
    push!(val_losses, val_loss)
    println("Epoch $epoch, training loss: $(train_loss), validation loss: $(val_loss)")
    if epoch % learning_rate_steps == 0 # adjust learning rate
        new_learning_rate = learning_rate * learning_rate_decay^(epoch/nepochs)
        Flux.adjust!(opt_state, new_learning_rate)
    end
end

#
# Analysis of the model
#

# create convenience functions for one station and selected times
function predict(model,wind_speed,wind_dir,nlags)
    # TODO we're not keeping the metadata of the model in a structured way yet.
    # continue UGGLY for now
    output_names=get_names(swh) # this looks in the global scope
    output_longitudes = copy(get_longitudes(swh))
    output_latitudes = copy(get_latitudes(swh))
    # get data and scale
    wind_times = get_times(wind_speed)
    n_times= length(wind_times)
    itraining_times = nlags:n_times
    ntraining_times = length(itraining_times)
    n_input_stations = length(get_names(wind_speed))
    # wind input
    u10_values = get_values(wind_speed)
    udir_values = get_values(wind_dir)
    n_input_vars=2 # wind_x, wind_y
    # convert wind to east and north components, wind direction is the direction from which the wind blows counting from north to east
    wind_x = u10_values .* -sind.(udir_values)
    wind_y = u10_values .* -cosd.(udir_values)
    # scale the wind speed to be of order 1
    wind_x = wind_x ./ wind_scale
    wind_y = wind_y ./ wind_scale  
    # convert wind to stress components
    for i in eachindex(wind_x)
        wind_x[i], wind_y[i] = uv_to_stress_xy(wind_x[i], wind_y[i])
    end
    # create input data
    x_input=zeros(Float32,nlags,n_input_stations*n_input_vars,ntraining_times)
    for itime in itraining_times
        wind_x_block=wind_x[:,itime-nlags+1:itime]
        wind_y_block=wind_y[:,itime-nlags+1:itime]
        x_block=Float32.(vcat(wind_x_block,wind_y_block))'
        x_input[:,:,(itime-nlags+1)].=x_block
    end
    # reserve memory for output
    output_times = copy(wind_times)
    output_sources = "time-series ai model"
    output_quantity = "wave_height"
    #output_names = TODO
    n_output_stations = length(output_names)
    # output_longitudes = copy(get_longitudes(wind_speed))
    # output_latitudes = copy(get_latitudes(wind_speed))
    y_hat = zeros(Float32,n_output_stations,n_times)
    y_hat[:,1:nlags-1] .= NaN # first nlags-1 time steps cannot be predicted
    # run model for each station
    for istation in 1:n_output_stations
        x_station = Flux.onehotbatch(fill(istation,ntraining_times),1:n_output_stations)
        y_hat[istation,nlags:end] .= wave_scale .* model((x_station, x_input))[1,:]
    end
    y_predicted = TimeSeries(y_hat, output_times, output_names, output_longitudes, output_latitudes, output_quantity, output_sources)
    return y_predicted
end

# compute predictions for the full time series
swh_predicted = predict(model,u10,udir,nlags)

#
# Analysis of the model
#
@info "Analysis of the trained model"

# If model was trained on the GPU, move it to the CPU
model_cpu=model |> cpu

# Create a clean directory for the model output and diagnostics
if isdir(model_name)
    rm(model_name,recursive=true,force=true)
end
mkdir(model_name)

# Save the model to BSON and JLD2 files
save(joinpath(model_name,"$(model_name).jld2"),"model",model_cpu)
BSON.@save joinpath(model_name,"$(model_name).bson") model_cpu


# Losses plot
plot(train_losses, label="training loss", xlabel="Epoch", ylabel="Loss")
plot!(val_losses, label="validation loss")
savefig(joinpath(model_name,"losses_train_val.png"))

# save the predicted time series
write_to_jld2(swh_predicted, joinpath(model_name,"predicted_wave_heights.jld2"))
write_to_netcdf(swh_predicted, joinpath(model_name,"predicted_wave_heights.nc"))

#
# Compute statistics per station
#
function stats_skipnan(y_true::TimeSeries,y_pred::TimeSeries)
    # check times and locations are the same
    @assert get_times(y_true) == get_times(y_pred) "Times of true and predicted series are not the same"
    @assert get_names(y_true) == get_names(y_pred) "Names of true and predicted series are not the same"
    names=get_names(y_true)
    # compute RMSE skipping NaNs
    y_true_values=copy(get_values(y_true))
    y_pred_values=get_values(y_pred)
    res=y_pred_values-y_true_values
    count_notnan = sum(isnan.(res).==false, dims=2)
    res[isnan.(res)] .= 0.0f0 # set NaNs to 0.0 
    y_true_values[isnan.(y_true_values)] .= 0.0f0 # set NaNs to 0.0
    relative_res=res ./ max.(y_true_values,0.1f0) # avoid division by zero or very small values
    bias=sum(res, dims=2)./ count_notnan
    rmse=sqrt.(sum(res.^2, dims=2) ./ count_notnan)
    mae=sum(abs.(res), dims=2)./ count_notnan
    relative_bias=sum(relative_res, dims=2)./ count_notnan
    scatter_index=sqrt.(sum(relative_res.^2, dims=2) ./ count_notnan)
    # create a DataFrame with the results
    stats=DataFrame(station_name=names,
        bias=vec(bias),
        rmse=vec(rmse),
        mae=vec(mae),
        relative_bias=vec(relative_bias),
        scatter_index=vec(scatter_index),
        count=vec(count_notnan))
    return stats
end

function plot_series(y_true::TimeSeries,y_pred::TimeSeries,station_name)
    quantity=replace(get_quantity(y_true),"_"=>" ")
    y_true_station=select_location_by_name(y_true,station_name)
    y_pred_station=select_location_by_name(y_pred,station_name)
    times=get_times(y_true_station)
    y_true_values=get_values(y_true_station)
    y_pred_values=get_values(y_pred_station)
    p=plot(times,vec(y_pred_values),label="predicted",color=:blue,xlabel="Time",ylabel=quantity,title="Station: $(station_name)")
    plot!(p,times,vec(y_true_values),label="target",color=:black)
    return p
end

function plot_all_series(y_true::TimeSeries,y_pred::TimeSeries,output_dir,filename_prefix)
    names=get_names(y_true)
    for name in names
        p=plot_series(y_true,y_pred,name)
        filename=joinpath(output_dir,"$(filename_prefix)_$(name).png")
        savefig(p,filename)
    end
end

function plot_scatter(y_true::TimeSeries,y_pred::TimeSeries,station_name)
    quantity=replace(get_quantity(y_true),"_"=>" ")
    y_true_station=select_location_by_name(y_true,station_name)
    y_pred_station=select_location_by_name(y_pred,station_name)
    times=get_times(y_true_station)
    y_true_values=get_values(y_true_station)
    y_pred_values=get_values(y_pred_station)
    p=scatter(vec(y_true_values),vec(y_pred_values),label=false,color=:black,ms=1,xlabel="target $(quantity)",ylabel="predicted $(quantity)",title="Scatter : $(station_name)")
    return p
end

function plot_all_scatter(y_true::TimeSeries,y_pred::TimeSeries,output_dir,filename_prefix)
    names=get_names(y_true)
    for name in names
        p=plot_scatter(y_true,y_pred,name)
        filename=joinpath(output_dir,"$(filename_prefix)_$(name).png")
        savefig(p,filename)
    end
end

timespans=Dict(
    "training"=>(DateTime(2021,1,1),DateTime(2023,12,31,23)),
    "test"=>(DateTime(2024,1,1),DateTime(2024,11,1)),
    "202401"=>(DateTime(2024,1,1),DateTime(2024,2,1,0)),
)

# compute average statistics over all stations
function average_stats(previous_stats,stats::DataFrame, timespan_name::String)
    nstations=size(stats,1)
    avg_bias=mean(stats.bias)
    avg_rmse=mean(stats.rmse)
    avg_mae=mean(stats.mae)
    avg_relative_bias=mean(stats.relative_bias)
    avg_scatter_index=mean(stats.scatter_index)
    stats_df = DataFrame(
        timespan=timespan_name,
        avg_bias=avg_bias,
        avg_rmse=avg_rmse,
        avg_mae=avg_mae,
        avg_relative_bias=avg_relative_bias,
        avg_scatter_index=avg_scatter_index,
        nstations=nstations
    )
    if previous_stats===nothing
        return stats_df
    else
        return vcat(previous_stats, stats_df)
    end
end


avg_stats_df=nothing
for (timespan_name,(tstart,tend)) in timespans
    @info "Computing statistics for timespan: $(timespan_name) ($(tstart) to $(tend))"
    swh_timespan=select_timespan(swh,tstart,tend)
    swh_predicted_timespan=select_timespan(swh_predicted,tstart,tend)
    stats=stats_skipnan(swh_timespan,swh_predicted_timespan)
    avg_stats_df=average_stats(avg_stats_df,stats,timespan_name)
    # Save the RMSEs into a DataFrame
    CSV.write(joinpath(model_name,"$(timespan_name)_statistice_wave_height.csv"),stats)
    @show stats
    # plot all series
    plot_all_series(swh_timespan,swh_predicted_timespan,model_name,"$(timespan_name)_wave_height")
    # plot all scatter plots
    plot_all_scatter(swh_timespan,swh_predicted_timespan,model_name,"$(timespan_name)_scatter_wave_height")
end
# Save the average statistics into a CSV file
@show avg_stats_df
CSV.write(joinpath(model_name,"average_statistics_wave_height.csv"),avg_stats_df)