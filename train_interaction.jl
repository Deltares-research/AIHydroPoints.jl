#train_interaction.jl
#
# Model for tide-surge interaction 

# Move to this folder if not already there
cd(@__DIR__)
# activate the environment
using Pkg
Pkg.activate(".")

#settings
# nchannel=(32,32,32,1)
# nchannel=(64,64,64,32,1)
nchannel=(128,64,64,64,32,1)
nlayers=length(nchannel) #number of layers
const nlags=64 #16 #time-steps used including the current time step
    # must be a power of 2
const nepochs=[200] #[50]
const nbatch=1024 #was 256
const learning_rate=1.0e-3
const regularization_weight=0.5e-5 #1.0e-4
# const model_name="surge_model_317stations_$(nlayers)sl_3yr_$(nbatch)batch_0p000005reg_$(n;ags)lags_$(nepochs[1])epochs"
const model_name="interaction_model_5stations_$(nlayers)sl_3yr_$(nbatch)batch_0p000005reg_$(nlags)lags_$(nepochs[1])epochs"

labels=["training","validation","testing"]
filenames=Dict()
#const nstations=5 #number of stations
# use the surge model residuals - 5stations
filenames["training"]=Dict(
    "waterlevel"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_training_residual.jld2",
    "tide"=>"tide_model_5stations_3tl_3yr_1024batch_0p0001reg_64nodes_10epochs_training_tide.jld2",
    "surge"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_training_surge.jld2")
filenames["validation"]=Dict(
    "waterlevel"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_testing_residual.jld2",
    "tide"=>"tide_model_5stations_3tl_3yr_1024batch_0p0001reg_64nodes_10epochs_testing_tide.jld2",
    "surge"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_testing_surge.jld2") #TODO trick
filenames["testing"]=Dict(
    "waterlevel"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_testing_residual.jld2",
    "tide"=>"tide_model_5stations_3tl_3yr_1024batch_0p0001reg_64nodes_10epochs_testing_tide.jld2",
    "surge"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_testing_surge.jld2") #TODO trick

# use surge residuals - 317 stations
# filenames["training"]=Dict(
#     "waterlevel"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_training_residual.jld2",
#     "tide"=>"tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_100epochs_training_tide.jld2",
#     "surge"=>"surge_model_317stations_4sl_3yr_256batch_0p000005reg_64nodes_50epochs_training_surge.jld2")
# filenames["validation"]=Dict(
#     "waterlevel"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_testing_residual.jld2",
#     "tide"=>"tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_100epochs_testing_tide.jld2",
#     "surge"=>"surge_model_317stations_4sl_3yr_256batch_0p000005reg_64nodes_50epochs_testing_surge.jld2") #TODO trick
# filenames["testing"]=Dict(
#     "waterlevel"=>"surge_model_5stations_4sl_3yr_256batch_0p000005reg_64nodes_20epochs_testing_residual.jld2",
#     "tide"=>"tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_100epochs_testing_tide.jld2",
#     "surge"=>"surge_model_317stations_4sl_3yr_256batch_0p000005reg_64nodes_50epochs_testing_surge.jld2") #TODO trick

# Force the use of the CPU
force_cpu=true #false

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
    data_surge=load(filenames[label]["surge"])
    data_tide=load(filenames[label]["tide"])
    # shorten the data where needed
    ntimes=size(data_surge["waterlevel"],2) # output steps
    nother=length(data_surge["times"])
    ndiff=nother-ntimes
    if ntimes<nother
        data_surge["times"]=data_surge["times"][ndiff+1:end]
    end
    nother=length(data_waterlevel["times"])
    ndiff=nother-ntimes
    if ntimes<nother
        data_waterlevel["times"]=data_waterlevel["times"][ndiff+1:end]
    end
    nother=length(data_tide["times"])
    ndiff=nother-ntimes
    if ntimes<nother
        data_tide["times"]=data_tide["times"][ndiff+1:end]
    end
    nother=size(data_waterlevel["waterlevel"],2)
    ndiff=nother-ntimes
    if ntimes<nother
        data_waterlevel["waterlevel"]=data_waterlevel["waterlevel"][:,ndiff+1:end]
    end
    nother=size(data_tide["waterlevel"],2)
    ndiff=nother-ntimes
    if ntimes<nother
        data_tide["waterlevel"]=data_tide["waterlevel"][:,ndiff+1:end]
    end
    # collect in Dict
    data[label]=Dict("waterlevel"=>data_waterlevel,"surge"=>data_surge, "tide"=>data_tide)
end

# prepare the data for training
function prepare_interaction_data(data_waterlevel,data_surge,data_tide,nlags)
    #time
    times = data_waterlevel["times"]
    surge_times = data_surge["times"]
    if !all(times .== surge_times)
        error("Times of waterlevel and surge data do not match")
    end
    itraining_times = (nlags):length(times)
    ntraining_times = length(itraining_times)
    # stations
    station_names = data_waterlevel["station_names"]
    nstations = length(station_names)
    # get data and scale
    waterlevel = data_waterlevel["waterlevel"]
    surge = data_surge["waterlevel"]
    tide = data_tide["waterlevel"] # TODO scaling needed?
    # The first input to the model is a one-hot encoding of the station index
    # It's a list of all combinations of stations and training times 
    station_index = collect(1:length(station_names))*ones(ntraining_times)'
    x_station = Flux.onehotbatch(station_index[:], 1:nstations)
    # Create list of all combinations of stations and training times
    all_times= [itime for i in 1:length(station_names), itime in itraining_times][:]
    # create input data
    x_tidesurge=zeros(Float32,nlags,2,nstations*ntraining_times)
    for itime in itraining_times
        surge_block=surge[:,itime-nlags+1:itime]'
        tide_block=tide[:,itime-nlags+1:itime]'
        for istation in 1:nstations # we need a copy of the data for each station
            isample=((itime-nlags)*nstations+istation)
            # size nlags,2,nstations*ntraining_times
            x_tidesurge[:,1,isample].=surge_block[:,istation]
            x_tidesurge[:,2,isample].=tide_block[:,istation]
        end
    end
    # create output data
    y_waterlevel=reshape(waterlevel[:,nlags:end],1,:) #flatten the waterlevel to a vector
    return (x_station, x_tidesurge, y_waterlevel)
end

# some meta data
station_names=data["training"]["waterlevel"]["station_names"]
nstations=length(station_names)
npars=2 # 2 input series

# prepare data for training
@info "Preparing data for training"
training_x_station,training_x_tidesurge,training_y = 
    prepare_interaction_data(data["training"]["waterlevel"],
    data["training"]["surge"],data["training"]["tide"],nlags)
training_x_station_gpu=training_x_station |> device
training_x_tidesurge_gpu=training_x_tidesurge |> device
training_y_gpu=training_y |> device

# prepare data for validation
@info "Preparing data for validation"
validation_x_station,validation_x_tidesurge,validation_y = 
    prepare_interaction_data(data["validation"]["waterlevel"],
    data["validation"]["surge"],data["validation"]["tide"],nlags)
# validation_x_station_gpu=validation_x_station |> device
# validation_x_tidesurge_gpu=validation_x_tidesurge |> device
# validation_y_gpu=validation_y |> device

# prepare data for testing
# @info "Preparing data for testing"
# testing_x_station,testing_x_tidesurge,testing_y = 
#     prepare_interaction_data(data["testing"]["waterlevel"],
#     data["testing"]["surge"],data["testing"]["tide"],nlags)
# testing_x_station_gpu=testing_x_station |> device
# testing_x_tidesurge_gpu=testing_x_tidesurge |> device
# testing_y_gpu=testing_y |> device
#testing_y_gpu=testing_y |> device

@info "Creating DataLoader"
dataloader = Flux.DataLoader((training_x_station_gpu,training_x_tidesurge_gpu,training_y_gpu), batchsize=nbatch, shuffle=true)


#
# Custom input layer
#
struct InteractionInputLayer{T}
    station_params::T
end

#constructor
InteractionInputLayer(nstations,nlags,npars) = InteractionInputLayer(
    Dense(nstations=>(nlags*npars),identity;bias=false)
)

# define the forward pass
function (l::InteractionInputLayer)(x)
    x_station, x_tidesurge = x
    nlags,npars,nbatch= size(x_tidesurge)
    s1 = l.station_params(x_station) 
    # output is -(nlags*npars,nbatch)
    # reshape to (nlags,npars,nbatch)
    s1 = reshape(s1, (nlags, npars, nbatch))
    z1 = s1 .* x_tidesurge
    return z1
end

# model for nlags=16
# model = Chain( 
#     InteractionInputLayer(nstations,nlags,npars),
#     Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#     Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#     Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
#     Conv((2,),nchannel[3]=>nchannel[4],identity,stride=(2,)),
#     Flux.flatten
# )
# model for nlags=32
# model = Chain(
#     InteractionInputLayer(nstations,nlags,npars),
#     Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
#     Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
#     Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
#     Conv((2,),nchannel[3]=>nchannel[4],relu,stride=(2,)),
#     Conv((2,),nchannel[4]=>nchannel[5],identity,stride=(2,)),
#     Flux.flatten
# )
# model for nlags=64
model = Chain(
    InteractionInputLayer(nstations,nlags,npars),
    Conv((2,),npars=>nchannel[1],relu,stride=(2,)),
    Conv((2,),nchannel[1]=>nchannel[2],relu,stride=(2,)),
    Conv((2,),nchannel[2]=>nchannel[3],relu,stride=(2,)),
    Conv((2,),nchannel[3]=>nchannel[4],relu,stride=(2,)),
    Conv((2,),nchannel[4]=>nchannel[5],relu,stride=(2,)),
    Conv((2,),nchannel[5]=>nchannel[6],identity,stride=(2,)),
    Flux.flatten
)   

model_gpu=model |> device # move the model to the GPU if available, or keep it on the CPU

function compute_loss(model,x_station,x_tidesurge,y)
    y_hat = model((x_station, x_tidesurge))
    return sqrt(Flux.mse(y_hat, y))
end

# ff=compute_loss(model_gpu,training_x_station_gpu,training_x_tidesurge_gpu,training_y_gpu)

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
        for (x_station, x_tidesurge, y) in dataloader
            dloss, grads = Flux.withgradient(model_gpu) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m((x_station, x_tidesurge))
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
        train_loss=compute_loss(model_cpu,training_x_station,training_x_tidesurge,training_y)
        push!(train_losses, train_loss)
        @info "compute validation loss"
        # val_loss=compute_loss(model_gpu,validation_x_station_gpu,validation_x_tidesurge_gpu,validation_y_gpu)
        val_loss=compute_loss(model_cpu,validation_x_station,validation_x_tidesurge,validation_y)
        push!(val_losses, val_loss)
        println("Epoch $epoch, training loss: $(train_loss), validation loss: $(val_loss)")
    end
end

#
# Analysis of the model
#

# create convenience functions for one station and selected times
function predict(model,data_tide,data_surge,istation,nstations,nlags)
    times = data_tide["times"]
    ntimes= length(times)
    itraining_times = nlags:ntimes
    ntraining_times = length(itraining_times)
    # station input
    x_station = Flux.onehotbatch(fill(istation,ntraining_times),1:nstations)
    # tidesurge input
    tide=data_tide["waterlevel"]
    surge=data_surge["waterlevel"]
    x_tidesurge=zeros(Float32,nlags,2,ntraining_times)
    for itime in itraining_times
        tide_block=tide[istation,itime-nlags+1:itime]
        surge_block=surge[istation,itime-nlags+1:itime]
        x_tidesurge[:,1,(itime-nlags+1)].=surge_block[:]
        x_tidesurge[:,2,(itime-nlags+1)].=tide_block[:]
    end
    y_hat = model((x_station, x_tidesurge))
    return y_hat[:]
end

# debug prediction
# ff_interaction=predict(model,data["training"]["tide"],data["training"]["surge"],1,nstations,nlags)

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


# Compute RMSE per station
function plot_series(model,data,model_name,prefix,itimes=nothing)
    rmses=[]
    if itimes==nothing
        itimes = 1:length(data["tide"]["times"])
    end
    # filter the times for <nlags
    itimes = itimes[itimes .>= nlags]
    itimes_base1 = itimes .-nlags.+1
    times = data["tide"]["times"]
    station_names = data["waterlevel"]["station_names"]
    waterlevel = data["waterlevel"]["waterlevel"]
    nstations = length(station_names)
    for istation=1:nstations
        station_name=station_names[istation]
        # function predict(model,data_wind,istation,nstations,nlags)
        interaction=predict(model,data["tide"],data["surge"],istation,nstations,nlags)
        rmse= sqrt(mean((interaction[itimes_base1]-waterlevel[istation,itimes]).^2))
        push!(rmses,rmse)
        # Plot measured and predicted interaction and the difference
        p1=plot(times[itimes],waterlevel[istation,itimes],label="Measured",
            xlabel="Time",ylabel="Waterlevel",title="Station $(station_name) RMSE=$(rmse)")
        p2=plot(times[itimes],interaction[itimes_base1],label="Predicted interaction")
        p3=plot(times[itimes],waterlevel[istation,itimes]-interaction[itimes_base1],label="Difference")
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

