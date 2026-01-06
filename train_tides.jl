# learn_tides.jl
#
# Train a model to predict tides on a set of timeseries.

# Move to this folder if not already there
cd(@__DIR__)
# activate the local environment
using Pkg
Pkg.activate(".")

# settings
#freqs=["M2","S2","K1","O1","H","P","N"] # with nodal frequencies
#freqs=["M2","S2","K1","O1","H"] # without nodal frequencies
#freqs=["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H","P","N"]
#freqs=["SSA","K1","O1","Q1","M2","S2","N2","H"] #prev
freqs=["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H"]
const nfreqs = length(freqs)
# Define the model
const n1 = 64 # number of outputs of the first layer
const n2 = 64 # number of outputs of the second layer
const nepochs = [100] #[100]
const nbatch = 1024 #was 256
#const learning_rate = 1.0e-4
const learning_rate = 1.0e-3
const n_layers = 3 # 0,1,2 or 3
const regularization_weight = 1.0e-4
const nobs_per_hour = 6 # number of observations per hour
#model_name="tide_model_317stations_$(n_layers)tl_3yr_$(nbatch)batch_0p0001reg_$(n1)nodes_$(nepochs[1])epochs"
model_name="tide_model_317stations_$(n_layers)tl_3yr_$(nbatch)batch_0p0001reg_$(n1)nodes_$(nepochs[1])epochs"

# datasets
# training_file="DCSM-FM_0_5nm_2008_3yr_5stations_his.jld2" #3years
# testing_file="DCSM-FM_0_5nm_2011_5stations_his.jld2"
training_file="DCSM-FM_0_5nm_2008_3yr_317stations_his.jld2" #3years
testing_file="DCSM-FM_0_5nm_2011_317stations_his.jld2"
# training_file="DCSM-FM_0_5nm_2008_3yr_50stations_his.jld2" #3years
# testing_file="DCSM-FM_0_5nm_2011_50stations_his.jld2"

# Force the use of the CPU
force_cpu=false

# Load required packages
using Dates
using Plots
using JLD2
using CUDA, cuDNN
using Flux
using ProgressMeter
using BSON
using Statistics
using CSV
using DataFrames
using NetCDF
include("tide_time.jl")
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


# prepare the data for training
function prepare_data(data)
    times = data["times"]
    station_names = data["station_names"]
    waterlevel = data["waterlevel"]
    ntimes = length(times)
    nstations = length(station_names)
    # The first inputto the model is a one-hot encoding of the station index 
    station_index = collect(1:length(station_names))*ones(ntimes)'
    x_station = Flux.onehotbatch(station_index[:], 1:nstations)
    # The second input to the model is the doodson phases of the main tidal constituents at the given times
    frequencies=primary_frequencies_as_doodson(freqs)
    nfreqs = size(frequencies,2)
    all_times= [time for i in 1:length(station_names), time in times] # take a subset of the data
    doodson = (get_doodson_eqvals(all_times[:])*frequencies)'
    x_doodson = Float32.(vcat(cos.(doodson),sin.(doodson)))
    y_waterlevel=reshape(waterlevel,1,:) #flatten the waterlevel to a vector
    return (x_station, x_doodson, y_waterlevel)
end

# Load the data
training_data = load(training_file)
testing_data = load(testing_file)
# derived data
nstations = length(training_data["station_names"])
# Prepare the data
#prepared_training_data = prepare_data(training_data)
training_x_station,training_x_doodson,training_y = prepare_data(training_data)
training_x_station_gpu=training_x_station |> device
training_x_doodson_gpu=training_x_doodson |> device
training_y_gpu=training_y |> device
#prepared_testing_data = prepare_data(testing_data) 
testing_x_station,testing_x_doodson,testing_y = prepare_data(testing_data)
testing_x_station_gpu=testing_x_station |> device
testing_x_doodson_gpu=testing_x_doodson |> device
testing_y_gpu=testing_y |> device

# Create a DataLoader
# dataloader = Flux.DataLoader((prepared_training_data[1] |>device, prepared_training_data[2] |>device, prepared_training_data[3] |>device), batchsize=nbatch, shuffle=true)
dataloader = Flux.DataLoader((training_x_station_gpu,training_x_doodson_gpu,training_y_gpu), batchsize=nbatch, shuffle=true)


#
# Custom input layer
#
struct TideInputLayer{T}
    station_params1::T
    doodson_params1::T
    station_params2::T
    doodson_params2::T
end
#constructor
TideInputLayer(nstations,nfreqs,n1) = TideInputLayer(
    Dense(nstations=>n1,identity;bias=false),
    Dense((2*nfreqs)=>n1,identity;bias=false),
    Dense(nstations=>n1,identity;bias=false),
    Dense((2*nfreqs)=>n1,identity;bias=false)
)
# define the forward pass
function (l::TideInputLayer)(x)
    x_station, x_doodson = x
    s1 = l.station_params1(x_station)
    d1 = l.doodson_params1(x_doodson)
    z1 = s1 .* d1
    # s2 = l.station_params2(x_station)
    # d2 = l.doodson_params2(x_doodson)
    # x2 = 0.1f0 .* s2 .* d2
    return (z1,z1)
end


#
# Custom tide layer
#
struct TideLayer{T}
    direct::T
    for_product::T
end
#constructor
TideLayer(n1_input,n1_output,n2_input,n2_putput) = TideLayer(Dense(n1_input,n1_output,relu),Dense(n2_input,n2_putput,relu))
# define the forward pass
function (l::TideLayer)(x)
    x1,x2 = x
    #r1 = l.direct(x1)
    r2 = l.for_product(x2).*x1
    #return (r1, r1.*r2)
    return x1 .+ r2,r2 # resnet with product
end

if n_layers==0
    model = Chain( # no tide layers
        TideInputLayer(nstations,nfreqs,n1),
        x->sum(x[1],dims=1)./n2
    ) 
elseif n_layers==1
    model = Chain( # one tide layer
        TideInputLayer(nstations,nfreqs,n1),
        TideLayer(n1,n2,n1,n2),
        x->sum(x[1],dims=1)./n2
    ) 
elseif n_layers==2
    model = Chain( # two tide layers
        TideInputLayer(nstations,nfreqs,n1),
        TideLayer(n1,n2,n1,n2),
        TideLayer(n1,n2,n1,n2),
        x->sum(x[1],dims=1)./n2
    )
elseif n_layers==3
    model = Chain( # three tide layers
        TideInputLayer(nstations,nfreqs,n1),
        TideLayer(n1,n2,n1,n2),
        TideLayer(n1,n2,n1,n2),
        TideLayer(n1,n2,n1,n2),
        x->sum(x[1],dims=1)./n2
    )
end
# tide_model_0tl = Chain( # two tide layers
#     TideInputLayer(nstations,nfreqs,n1),
#     #TideLayer(n1,n2,n1,n2),
#     #TideLayer(n1,n2,n1,n2),
#     #Parallel(.+,Dense(n1,1),Dense(n1,1))
#     #Parallel(.+,Dense(n1,1),Dense(n1,1,zero))
#     x->sum(x[1],dims=1)./n2
# ) #|> gpu

model_gpu=model |> device # move the model to the GPU if available, or keep it on the CPU

function compute_loss(model,x_station,x_doodson,y)
    y_hat = model(x_station, x_doodson)
    return sqrt(Flux.mse(y_hat, y))
end

# training loop 
train_losses = []
test_losses = []
acc_losses = []
for nepoch in nepochs
    #opt_state = Flux.setup(Flux.Adam(learning_rate), model_gpu)
    opt_state = Flux.setup(OptimiserChain(WeightDecay(regularization_weight), Adam(learning_rate)), model_gpu)

    @showprogress for epoch in 1:nepoch
        loss=0.0f0
        for (x_station, x_doodson, y) in dataloader
            dloss, grads = Flux.withgradient(model_gpu) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x_station, x_doodson)
                Flux.mse(y_hat, y)
            end
            Flux.update!(opt_state, model_gpu, grads[1])
            loss += dloss
        end
        println("Epoch $epoch, accumulated loss: $(loss)")
        push!(acc_losses, loss)  # logging, outside gradient context
        train_loss=compute_loss(model_gpu,training_x_station_gpu,training_x_doodson_gpu,training_y_gpu)
        push!(train_losses, train_loss)
        test_loss=compute_loss(model_gpu,testing_x_station_gpu,testing_x_doodson_gpu,testing_y_gpu)
        push!(test_losses, test_loss)
        println("Epoch $epoch, training loss: $(train_loss), testing loss: $(test_loss)")
    end
end

#
# Analysis of the model
#

# create convenience functions for one station and selected times
function predict(model,times,istation,nstations)
    x_station = Flux.onehotbatch(fill(istation,length(times)), 1:nstations)
    frequencies=primary_frequencies_as_doodson(freqs)
    doodson = (get_doodson_eqvals(times)*frequencies)'
    x_doodson = Float32.(vcat(cos.(doodson),sin.(doodson)))
    y_hat = model(x_station, x_doodson)
    return y_hat[:]
end

# If model was trained on the GPU, move it to the CPU
model=model_gpu |> cpu

if !isdir(model_name)
    mkdir(model_name)
end

# Save the model
BSON.@save joinpath(model_name,"$(model_name).bson") model

# Plot the losses
plot(acc_losses, label="training loss", xlabel="Epoch", ylabel="Loss",yscale=:log10)
savefig(joinpath(model_name,"losses.png"))
istart=5
plot(istart:length(acc_losses),acc_losses[5:end], label="training loss", xlabel="Epoch", ylabel="Loss")
savefig(joinpath(model_name,"losses_starting_at_$(istart).png"))

# New losses plot
plot(train_losses, label="training loss", xlabel="Epoch", ylabel="Loss")
plot!(test_losses, label="testing loss")
savefig(joinpath(model_name,"losses_train_test.png"))
istart=5
plot(5:length(train_losses),train_losses[5:end], label="training loss", xlabel="Epoch", ylabel="Loss")
plot!(5:length(test_losses),test_losses[5:end], label="testing loss")
savefig(joinpath(model_name,"losses_train_test_starting_at_$(istart).png"))

# Compute RMSE per station
function plot_series(model,data,model_name,prefix,itimes=nothing)
    rmses=[]
    if itimes==nothing
        itimes = 1:length(data["times"])
    end
    times = data["times"][itimes]
    station_names = data["station_names"]
    waterlevel = data["waterlevel"]
    nstations = length(station_names)
    for istation=1:nstations
        station_name=station_names[istation]
        tide=predict(model,times,istation,nstations)
        rmse= sqrt(mean((tide-waterlevel[istation,itimes]).^2))
        push!(rmses,rmse)
        # Plot measured and predicted tides and the difference
        p1=plot(times,waterlevel[istation,itimes],label="Measured",
            xlabel="Time",ylabel="Waterlevel",title="Station $(station_name) RMSE=$(rmse)")
        p2=plot(times,tide,label="Predicted")
        p3=plot(times,waterlevel[istation,itimes]-tide,label="Difference")
        plot(p1,p2,p3,layout=(3,1))
        savefig(joinpath(model_name,"$(prefix)_station_$(station_name).png"))
    end
    return rmses
end
training_rmses=plot_series(model,training_data,model_name,"training")
testing_rmses=plot_series(model,testing_data,model_name,"testing")
selected_times_training=1:div(6*24*29,nobs_per_hour)
training_rmses_14d=plot_series(model,training_data,model_name,"training_14days",selected_times_training)
selected_times_testing=1:div(6*24*29,nobs_per_hour)
testing_rmses_14d=plot_series(model,testing_data,model_name,"testing_14days",selected_times_testing)

# Save the RMSEs into a DataFrame
rmses=DataFrame(station_name=training_data["station_names"],training=training_rmses,testing=testing_rmses)
CSV.write(joinpath(model_name,"rmses.csv"),rmses)

# Save the predictions and residuals into a netcdf file
function write_series(model,data,model_name,prefix,itimes=nothing)
    station_names = data["station_names"]
    nstations = length(station_names)
    predictions=zeros(Float32,nstations,length(data["times"]))
    residuals=zeros(Float32,nstations,length(data["times"]))
    if itimes==nothing # all times
        itimes = 1:length(data["times"])
    end
    times = data["times"][itimes]
    waterlevel = data["waterlevel"]
    for istation=1:nstations
        tide=predict(model,times,istation,nstations)
        predictions[istation,:]=tide
        residuals[istation,:]=waterlevel[istation,itimes].-tide
    end
    # write to netcdf
    filename_predictions=joinpath(model_name,"$(prefix)_tides.nc")
    filename_residuals=joinpath(model_name,"$(prefix)_surge.nc")
    station_x=Float64.(data["station_x_coordinate"])
    station_y=Float64.(data["station_y_coordinate"])
    @show station_names
    #waterlevel_series_to_netcdf(filename, times, waterlevels, station_names,station_x,station_y)
    waterlevel_series_to_netcdf(filename_predictions, times, predictions, station_names,station_x,station_y)
    waterlevel_series_to_netcdf(filename_residuals, times, residuals, station_names,station_x,station_y)
    return predictions,residuals
end
training_predictions,training_residuals=write_series(model,training_data,model_name,"training")
testing_predictions,testing_residuals=write_series(model,testing_data,model_name,"testing")

# Save the model as a JLD2 file
# training period
save(joinpath(model_name,"$(model_name)_training_tide.jld2"),Dict(
"station_x_coordinate"=>training_data["station_x_coordinate"],
"station_y_coordinate"=>training_data["station_y_coordinate"],
"station_names"=>training_data["station_names"],
"times"=>training_data["times"],
"waterlevel"=>training_predictions)) 
save(joinpath(model_name,"$(model_name)_training_surge.jld2"),Dict(
"station_x_coordinate"=>training_data["station_x_coordinate"],
"station_y_coordinate"=>training_data["station_y_coordinate"],
"station_names"=>training_data["station_names"],
"times"=>training_data["times"],
"waterlevel"=>training_residuals))
# testing period 
save(joinpath(model_name,"$(model_name)_testing_tide.jld2"),Dict(
"station_x_coordinate"=>testing_data["station_x_coordinate"],
"station_y_coordinate"=>testing_data["station_y_coordinate"],
"station_names"=>testing_data["station_names"],
"times"=>testing_data["times"],
"waterlevel"=>testing_predictions)) 
save(joinpath(model_name,"$(model_name)_testing_surge.jld2"),Dict(
"station_x_coordinate"=>testing_data["station_x_coordinate"],
"station_y_coordinate"=>testing_data["station_y_coordinate"],
"station_names"=>testing_data["station_names"],
"times"=>testing_data["times"],
"waterlevel"=>testing_residuals)) 
