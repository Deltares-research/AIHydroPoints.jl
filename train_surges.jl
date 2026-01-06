#learn_surges.jl
#
# Train a model to predict surges from wind stress and air pressure

# Move to this folder if not already there
cd(@__DIR__)
# activate the environment
using Pkg
Pkg.activate(".")


#settings
nchannel=(32,32,32,1)
nlayers=length(nchannel) #number of layers
const nlags=16 #time-steps used including the current time step
    # must be a power of 2
const nepochs=[20]
const nbatch=256 #was 256
const learning_rate=1.0e-3
const regularization_weight=0.5e-5 #1.0e-4
# const model_name="surge_model_317stations_$(nlayers)sl_3yr_$(nbatch)batch_0p000005reg_$(n;ags)lags_$(nepochs[1])epochs"
const model_name="surge_model_5stations_$(nlayers)sl_3yr_$(nbatch)batch_0p000005reg_$(nlags)lags_$(nepochs[1])epochs"

labels=["training","validation","testing"]
filenames=Dict()
# filenames["training"]=Dict(
#     "waterlevel"=>"DCSM-FM_0_5nm_2008_3yr_5stations_his.jld2",
#     "wind"=>"era5_wind_stress_2008_training.jld2")
# filenames["validation"]=Dict(
#     "waterlevel"=>"DCSM-FM_0_5nm_2012_5stations_his.jld2",
#     "wind"=>"era5_wind_stress_2012_validation.jld2")
# filenames["testing"]=Dict(
#     "waterlevel"=>"DCSM-FM_0_5nm_2011_5stations_his.jld2",
#     "wind"=>"era5_wind_stress_2011_testing.jld2")

## use the tide model residuals - 5stations
filenames["training"]=Dict(
    "waterlevel"=>"tide_model_5stations_3tl_3yr_1024batch_0p0001reg_64nodes_10epochs_training_surge.jld2",
    "wind"=>"era5_wind_stress_2008_training.jld2")
filenames["validation"]=Dict(
    "waterlevel"=>"tide_model_5stations_3tl_3yr_1024batch_0p0001reg_64nodes_10epochs_testing_surge.jld2",
    "wind"=>"era5_wind_stress_2011_testing.jld2") #TODO trick
filenames["testing"]=Dict(
    "waterlevel"=>"tide_model_5stations_3tl_3yr_1024batch_0p0001reg_64nodes_10epochs_testing_surge.jld2",
    "wind"=>"era5_wind_stress_2011_testing.jld2")

## use the tide model residuals - 317 stations
# filenames["training"]=Dict(
#     "waterlevel"=>"tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_100epochs_training_surge.jld2",
#     "wind"=>"era5_wind_stress_2008_training.jld2")
# filenames["validation"]=Dict(
#     "waterlevel"=>"tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_100epochs_testing_surge.jld2",
#     "wind"=>"era5_wind_stress_2011_testing.jld2") #TODO trick
# filenames["testing"]=Dict(
#     "waterlevel"=>"tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_100epochs_testing_surge.jld2",
#     "wind"=>"era5_wind_stress_2011_testing.jld2")
        

# Force the use of the CPU
force_cpu=false #false

# Load required packages
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
include("netcdf_utils.jl")

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

# load the data
@info "Loading data"
data=Dict()
for label in labels
    @show label
    data_waterlevel=load(filenames[label]["waterlevel"])
    data_wind=load(filenames[label]["wind"])
    data[label]=Dict("waterlevel"=>data_waterlevel,"wind"=>data_wind)
end

# if the waterlevel contains more times than the wind the shorten it
function shorten_waterlevel!(data_waterlevel, wind_times)
    waterlevel_times=data_waterlevel["times"]
    if waterlevel_times[1]!=wind_times[1]
        error("start of times for wind and waterevel are not matching")
    end
    ntimes=length(wind_times)
    if length(waterlevel_times)>ntimes
        data_waterlevel["times"]=data_waterlevel["times"][1:ntimes]
        data_waterlevel["waterlevel"]=data_waterlevel["waterlevel"][:,1:ntimes]
    end
    nothing # no output
end

for label in labels
    shorten_waterlevel!(data[label]["waterlevel"],data[label]["wind"]["times"])
end

#debugging
# data_waterlevel=data["training"]["waterlevel"]
# data_wind=data["training"]["wind"]

# prepare the data for training
function prepare_surge_data(data_waterlevel,data_wind,nlags)
    #time
    times = data_waterlevel["times"]
    wind_times = data_wind["times"]
    if !all(times .== wind_times)
        error("Times of waterlevel and wind data do not match")
    end
    itraining_times = (nlags):length(times)
    ntraining_times = length(itraining_times)
    # stations
    station_names = data_waterlevel["station_names"]
    nstations = length(station_names)
    nwind = length(data_wind["station_x_coordinate"])
    # get data and scale
    waterlevel = data_waterlevel["waterlevel"]
    press=2e-4.*(data_wind["pressure"].-1e5)
    stress_x=data_wind["stress_x"]
    stress_y=data_wind["stress_y"]
    # The first input to the model is a one-hot encoding of the station index
    # It's a list of all combinations of stations and training times 
    station_index = collect(1:length(station_names))*ones(ntraining_times)'
    x_station = Flux.onehotbatch(station_index[:], 1:nstations)
    # Create list of all combinations of stations and training times
    all_times= [itime for i in 1:length(station_names), itime in itraining_times][:]
    # create input data
    x_stress_press=zeros(Float32,nlags,nwind*3,nstations*ntraining_times)
    for itime in itraining_times
        stress_x_block=stress_x[:,itime-nlags+1:itime]
        stress_y_block=stress_y[:,itime-nlags+1:itime]
        press_block=press[:,itime-nlags+1:itime]
        x_block=Float32.(vcat(stress_x_block,stress_y_block,press_block))'
        for istation in 1:nstations # we need a copy of the data for each station
            isample=((itime-nlags)*nstations+istation)
            # @show itime,istation,isample
            # @show size(x_block), size(x_stress_press[:,:,isample])
            x_stress_press[:,:,isample].=x_block
        end
    end
    # create output data
    y_waterlevel=reshape(waterlevel[:,nlags:end],1,:) #flatten the waterlevel to a vector
    return (x_station, x_stress_press, y_waterlevel)
end


# some meta data
station_names=data["training"]["waterlevel"]["station_names"]
nstations=length(station_names)
npars=3*length(data["training"]["wind"]["station_x_coordinate"])

# prepare data for training
@info "Preparing data for training"
training_x_station,training_x_wind,training_y = 
    prepare_surge_data(data["training"]["waterlevel"],
    data["training"]["wind"],nlags)
training_x_station_gpu=training_x_station |> device
training_x_wind_gpu=training_x_wind |> device
training_y_gpu=training_y |> device

# prepare data for validation
validation_x_station,validation_x_wind,validation_y = 
    prepare_surge_data(data["validation"]["waterlevel"],
    data["validation"]["wind"],nlags)
# validation_x_station_gpu=validation_x_station |> device
# validation_x_wind_gpu=validation_x_wind |> device
# validation_y_gpu=validation_y |> device

# prepare data for testing
# testing_x_station,testing_x_wind,testing_y = 
#     prepare_surge_data(data["testing"]["waterlevel"],
#     data["testing"]["wind"],nlags)
# testing_x_station_gpu=testing_x_station |> device
# testing_x_wind_gpu=testing_x_wind |> device
# testing_y_gpu=testing_y |> device

# Create a DataLoader
@info "Creating DataLoader"
dataloader = Flux.DataLoader((training_x_station_gpu,training_x_wind_gpu,training_y_gpu), batchsize=nbatch, shuffle=true)


#
# Custom input layer
#
struct WindInputLayer{T}
    station_params::T
end

#constructor
WindInputLayer(nstations,nlags,npars) = WindInputLayer(
    Dense(nstations=>(nlags*npars),identity;bias=false)
)

# define the forward pass
function (l::WindInputLayer)(x)
    x_station, x_wind = x
    nlags,npars,nbatch= size(x_wind)
    s1 = l.station_params(x_station) 
    # output is -(nlags*npars,nbatch)
    # reshape to (nlags,npars,nbatch)
    s1 = reshape(s1, (nlags, npars, nbatch))
    z1 = s1 .* x_wind
    return z1
end

#
# Define the surge model
#
# if length(nchannel)==1
#     surge_model = Chain(
#         WindInputLayer(nstations,nlags,npars),
#         Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#         #Flux.flatten
#     )
# elseif length(nchannel)==2
#     surge_model = Chain(
#         WindInputLayer(nstations,nlags,npars),
#         Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#         Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#         #Flux.flatten
#     )
# elseif length(nchannel)==3
#     surge_model = Chain(
#         WindInputLayer(nstations,nlags,npars),
#         Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#         Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#         Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
#         #Flux.flatten
#     )
# elseif length(nchannel)==4
#     surge_model = Chain(
#         WindInputLayer(nstations,nlags,npars),
#         Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#         Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#         Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
#         Conv((2,),nchannel[3]=>nchannel[4],relu,stride=(2,)),
#         Flux.flatten
#     )
# end

model = Chain(
    WindInputLayer(nstations,nlags,npars),
    Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
    Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
    Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
    Conv((2,),nchannel[3]=>nchannel[4],identity,stride=(2,)),
    Flux.flatten
)

model_gpu=model |> device # move the model to the GPU if available, or keep it on the CPU

function compute_loss(model,x_station,x_wind,y)
    y_hat = model((x_station, x_wind))
    return sqrt(Flux.mse(y_hat, y))
end

#ff=compute_loss(model_gpu,training_x_station_gpu,training_x_wind_gpu,training_y_gpu)

# training loop 
@info "Training the model"
train_losses = []
val_losses = []
acc_losses = []
for nepoch in nepochs
    #opt_state = Flux.setup(Flux.Adam(learning_rate), model_gpu)
    opt_state = Flux.setup(OptimiserChain(WeightDecay(regularization_weight), Adam(learning_rate)), model_gpu)

    @showprogress for epoch in 1:nepoch
        @info "Epoch $(epoch)"
        loss=0.0f0
        for (x_station, x_wind, y) in dataloader
            dloss, grads = Flux.withgradient(model_gpu) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m((x_station, x_wind))
                Flux.mse(y_hat, y)
            end
            Flux.update!(opt_state, model_gpu, grads[1])
            loss += dloss
        end
        println("Epoch $epoch, accumulated loss: $(loss)")
        push!(acc_losses, loss)  # logging, outside gradient context
        @info "compute training loss"
        model_cpu=model_gpu |> cpu
        # train_loss=compute_loss(model_gpu,training_x_station_gpu,training_x_wind_gpu,training_y_gpu)
        train_loss=compute_loss(model_cpu,training_x_station,training_x_wind,training_y)
        push!(train_losses, train_loss)
        @info "compute validation loss"
        # val_loss=compute_loss(model_gpu,validation_x_station_gpu,validation_x_wind_gpu,validation_y_gpu)
        val_loss=compute_loss(model_cpu,validation_x_station,validation_x_wind,validation_y)
        push!(val_losses, val_loss)
        println("Epoch $epoch, training loss: $(train_loss), validation loss: $(val_loss)")
    end
end

#
# Analysis of the model
#

# create convenience functions for one station and selected times
function predict(model,data_wind,istation,nstations,nlags)
    wind_times = data_wind["times"]
    ntimes= length(wind_times)
    itraining_times = nlags:ntimes
    ntraining_times = length(itraining_times)
    nwind = length(data_wind["station_x_coordinate"])
    # station input
    x_station = Flux.onehotbatch(fill(istation,ntraining_times),1:nstations)
    # wind input
    stress_x=data_wind["stress_x"]
    stress_y=data_wind["stress_y"]
    press=2e-4.*(data_wind["pressure"].-1e5)
    x_stress_press=zeros(Float32,nlags,nwind*3,ntraining_times)
    for itime in itraining_times
        stress_x_block=stress_x[:,itime-nlags+1:itime]
        stress_y_block=stress_y[:,itime-nlags+1:itime]
        press_block=press[:,itime-nlags+1:itime]
        x_block=Float32.(vcat(stress_x_block,stress_y_block,press_block))'
        x_stress_press[:,:,(itime-nlags+1)].=x_block
    end
    y_hat = model((x_station, x_stress_press))
    return y_hat[:]
end

# debug prediction
# surge=predict(model,data["training"]["wind"],1,nstations,nlags)

#
# Analysis of the model
#
@info "Analysis of the trained model"

# If model was trained on the GPU, move it to the CPU
model=model_gpu |> cpu

if !isdir(model_name)
    mkdir(model_name)
end

# Save the model
BSON.@save joinpath(model_name,"$(model_name).bson") model


# Losses plot
plot(train_losses, label="training loss", xlabel="Epoch", ylabel="Loss")
plot!(val_losses, label="validation loss")
savefig(joinpath(model_name,"losses_train_val.png"))
# istart=5
# plot(5:length(train_losses),train_losses[5:end], label="training loss", xlabel="Epoch", ylabel="Loss")
# plot!(5:length(_losses),test_losses[5:end], label="testing loss")
# savefig(joinpath(model_name,"losses_train_test_starting_at_$(istart).png"))

# Compute RMSE per station
function plot_series(model,data,model_name,prefix,itimes=nothing)
    rmses=[]
    if itimes==nothing
        itimes = 1:length(data["wind"]["times"])
    end
    # filter the times for <nlags
    itimes = itimes[itimes .>= nlags]
    itimes_base1 = itimes .-nlags.+1
    times = data["wind"]["times"]
    station_names = data["waterlevel"]["station_names"]
    waterlevel = data["waterlevel"]["waterlevel"]
    nstations = length(station_names)
    for istation=1:nstations
        station_name=station_names[istation]
        # function predict(model,data_wind,istation,nstations,nlags)
        surge=predict(model,data["wind"],istation,nstations,nlags)
        rmse= sqrt(mean((surge[itimes_base1]-waterlevel[istation,itimes]).^2))
        push!(rmses,rmse)
        # Plot measured and predicted tides and the difference
        p1=plot(times[itimes],waterlevel[istation,itimes],label="Measured",
            xlabel="Time",ylabel="Waterlevel",title="Station $(station_name) RMSE=$(rmse)")
        p2=plot(times[itimes],surge[itimes_base1],label="Predicted surge")
        p3=plot(times[itimes],waterlevel[istation,itimes]-surge[itimes_base1],label="Difference")
        plot(p1,p2,p3,layout=(3,1))
        savefig(joinpath(model_name,"$(prefix)_station_$(station_name).png"))
    end
    return rmses
end
training_rmses=plot_series(model,data["training"],model_name,"training")
validation_rmses=plot_series(model,data["validation"],model_name,"validation")
# for short period
selected_times_training=1:24*15
training_rmses_14d=plot_series(model,data["training"],model_name,"training_14days",selected_times_training)
selected_times_testing=1:24*15
validation_rmses_14d=plot_series(model,data["validation"],model_name,"validation_14days",selected_times_testing)

# Save the RMSEs into a DataFrame
rmses=DataFrame(station_name=data["training"]["waterlevel"]["station_names"],training=training_rmses,validation=validation_rmses)
CSV.write(joinpath(model_name,"rmses.csv"),rmses)



# Save the predictions and residuals into a netcdf file
function write_surge_series(model,data,model_name,prefix,nlags)
    station_names = data["waterlevel"]["station_names"]
    nstations = length(station_names)
    times = data["waterlevel"]["times"]
    itimes = nlags:length(data["waterlevel"]["times"])
    itimes_base1 = itimes .-nlags.+1
    ntimes= length(itimes)
    times=times[itimes]
    # compute predictions and residuals
    predictions=zeros(Float32,nstations,ntimes)
    residuals=zeros(Float32,nstations,ntimes)
    waterlevel = data["waterlevel"]["waterlevel"]
    for istation=1:nstations
        surge=predict(model,data["wind"],istation,nstations,nlags)
        predictions[istation,:]=surge
        residuals[istation,:]=waterlevel[istation,itimes].-surge
    end
    # write to netcdf
    filename_predictions=joinpath(model_name,"$(prefix)_surge_prediction.nc")
    filename_residuals=joinpath(model_name,"$(prefix)_residual.nc")
    station_x=Float64.(data["waterlevel"]["station_x_coordinate"])
    station_y=Float64.(data["waterlevel"]["station_y_coordinate"])
    @show station_names
    #waterlevel_series_to_netcdf(filename, times, waterlevels, station_names,station_x,station_y)
    waterlevel_series_to_netcdf(filename_predictions, times, predictions, station_names,station_x,station_y)
    waterlevel_series_to_netcdf(filename_residuals, times, residuals, station_names,station_x,station_y)
    return predictions,residuals
end

training_predictions,training_residuals=write_surge_series(model,data["training"],model_name,"training",nlags)
validation_predictions,validation_residuals=write_surge_series(model,data["validation"],model_name,"validation",nlags)

# Save the model as a JLD2 file
# training period
save(joinpath(model_name,"$(model_name)_training_surge.jld2"),Dict(
"station_x_coordinate"=>data["training"]["waterlevel"]["station_x_coordinate"],
"station_y_coordinate"=>data["training"]["waterlevel"]["station_y_coordinate"],
"station_names"=>data["training"]["waterlevel"]["station_names"],
"times"=>data["training"]["waterlevel"]["times"],
"waterlevel"=>training_predictions)) 
save(joinpath(model_name,"$(model_name)_training_residual.jld2"),Dict(
"station_x_coordinate"=>data["training"]["waterlevel"]["station_x_coordinate"],
"station_y_coordinate"=>data["training"]["waterlevel"]["station_y_coordinate"],
"station_names"=>data["training"]["waterlevel"]["station_names"],
"times"=>data["training"]["waterlevel"]["times"],
"waterlevel"=>training_residuals))
#validation period
save(joinpath(model_name,"$(model_name)_testing_surge.jld2"),Dict(
"station_x_coordinate"=>data["validation"]["waterlevel"]["station_x_coordinate"],
"station_y_coordinate"=>data["validation"]["waterlevel"]["station_y_coordinate"],
"station_names"=>data["validation"]["waterlevel"]["station_names"],
"times"=>data["validation"]["waterlevel"]["times"],
"waterlevel"=>validation_predictions)) 
save(joinpath(model_name,"$(model_name)_testing_residual.jld2"),Dict(
"station_x_coordinate"=>data["validation"]["waterlevel"]["station_x_coordinate"],
"station_y_coordinate"=>data["validation"]["waterlevel"]["station_y_coordinate"],
"station_names"=>data["validation"]["waterlevel"]["station_names"],
"times"=>data["validation"]["waterlevel"]["times"],
"waterlevel"=>validation_residuals)) 
