# training.jl
# Functions to do training and saving and loading training runs
# Each new model type implements a settings type (subtype of AbstractModelSettings)
# and uses it to dispatch the various functions defined here.

using TOML
using JLD2
using Plots

"""
    AbstractModelSettings

Type that will store the various settings to build and train a model.
Will be subtyped to dispatch routines for the various models.
Any particular implementation of ModelSettings should include at minimum:
- `model_name`: Name of the model.
- `model_dir`: Path to directory where everything (model, validation prediction, plots, etc.) will be saved.
- `nepochs`: Number of epochs used during training.
- `nbatches`: Number of batches to split the training datat into.
- `learning_rate`: Learning rate of the Adam optimizer used.
- `weight_reg`: Weight Decay parameter
- `use_gpu`: Whether to train on gpu
"""
abstract type AbstractModelSettings end

"""
    save_settings(settings::AbstractModelSettings)

Saves settings to "settings.toml" file

# Arguments

- `settings::AbstractModelSettings`: settings to be saved
"""
function save_settings(settings::AbstractModelSettings)
    fn = joinpath(settings.model_dir, "settings.toml")
    # For convenience when loading we also write the type of the settings written
    tmp = Dict(key => getfield(settings, key) for key in propertynames(settings))
    dict = Dict(string(typeof(settings)) => tmp)
    open(fn, "w") do io
        TOML.print(io, dict)
    end
end

"""
    load_settings(fn)

Loads settings from file

# Arguments

- `fn`: path to settings.toml file
"""
function load_settings(fn)
    dict = TOML.parsefile(fn)
    # Deduce settings type
    settings_type = Symbol(only(keys(dict)))
    # Construct settings
    vals = only(values(dict))
    sett_type = @eval $settings_type
    settings = sett_type(; (Symbol.(keys(vals)) .=> values(vals))...)
    return settings
end

"""
    save_model(model, settings::AbstractModelSettings)

Saves model to "model.jld2" file

# Arguments

- `model`: Model to be saved.
- `settings::AbstractModelSettings`: Settings of the model, used to define save path
"""
function save_model(model, settings::AbstractModelSettings)
    fn = joinpath(settings.model_dir, "model.jld2")
    jldsave(fn, model_state=Flux.state(model))
end

"""
    load_model(settings::AbstractModelSettings, model_constructor)

Load model from JLD2 file

# Arguments

- `settings::AbstractModelSettings`: Settings of the model.
- `model_constructor`: Function that builds to model. Used to create a blank model into which parameters are loaded.
"""
function load_model(settings::AbstractModelSettings, model_constructor)
    model = model_constructor(settings.model_pars)
    model_state = JLD2.load(joinpath(settings.model_dir, "model.jld2"), "model_state")
    Flux.loadmodel!(model, model_state)

    return model
end

"""
    load_run(fn_dir, model_constructor)

Load a training run from directory. Creates the model settings struct and the trained model

# Arguments

- `fn_dir`: Path to run directory
- `model_constructor`: Function that builds to model. Used to create a blank model into which parameters are loaded. 
"""
function load_run(fn_dir, model_constructor)
    settings = load_settings(joinpath(fn_dir, "settings.toml"))
    model = load_model(settings, model_constructor)

    return model, settings
end

"""
    prepare_train_data(ts::TimeSeries, settings::AbstractModelSettings)

Create training data from a TimeSeries ts using the hyperparameters in settings.

# Arguments

- `ts::TimeSeries`: Input time series
- `settings::AbstractModelSettings`: settings containing hyperparameters
"""
function prepare_train_data(ts::TimeSeries, settings::AbstractModelSettings)
    error("Function prepare_data not defined for settings $(typeof(settings))")
end

"""
    prepare_inputs(data, settings::AbstractModelSettings)

Helper function to create just the inputs to a model from data.
Also useful during prediction

# Arguments

- `data`: data to create model inputs from
- `settings::AbstractModelSettings`: settings containing hyperparameters
"""
function prepare_inputs(data, settings::AbstractModelSettings)
    error("Function prepare_inputs not defined for settings $(typeof(settings))")    
end

"""
    compute_loss(model, settings::AbstractModelSettings, data)

Loss function for training and/or diagnostics during training

# Arguments

- `model`: model of which to evaluate performance
- `settings::AbstractModelSettings`: settings containing hyperparameters, used here for dispatch
- `data`: data used in model performance evaluation
"""
function compute_loss(model, settings::AbstractModelSettings, data)
    error("Function compute loss not implemented for settings $(typeof(settings))")
end

"""
    train_epoch!(model, settings::AbstractModelSettings, dataloader, opt_state)

Function training a model for a single epoch, based on the data in dataloader and the Optimizer state opt_state

# Arguments

- `model`: model to train
- `settings::AbstractModelSettings`: settings containing hyperparameters, used for dispatch
- `dataloader`: Flux DataLoader containing train data
- `opt_state`: Optimizer state
"""
function train_epoch!(model, settings::AbstractModelSettings, dataloader, opt_state)
    error("Function train_epoch! not implemented for settings $(typeof(settings))")    
end

"""
    train_model(model, settings::AbstractModelSettings, train_data, test_data)

Train model using train_data, evaluate performance using test_data
hyperparameters are stored in settings

To use this for a particular model type, implement the train_epoch! and compute_loss functions
for the relevant settings type

# Arguments

- `model`: model to train
- `settings::AbstractModelSettings`: settings containing hyperparameters, used for dispatching train_epoch!, compute_loss functions
- `train_data`: Prepared train data
- `test_data`: Prepared test data
"""
function train_model(model, settings::AbstractModelSettings, train_data, test_data)
    nepochs = settings.nepochs
    nbatches = settings.nbatches
    learning_rate = settings.learning_rate
    weight_reg = settings.weight_reg
    use_gpu = settings.use_gpu

    if use_gpu && CUDA.has_cuda()
        @info "Training on GPU"
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    train_losses = []
    test_losses = []
    acc_losses = []

    model = model |> device

    train_data = train_data |> device
    test_data = test_data |> device
    dataloader = Flux.DataLoader(train_data, batchsize=nbatches, shuffle=true)

    opt_state = Flux.setup(OptimiserChain(WeightDecay(weight_reg), Adam(learning_rate)), model)
    # opt_state = Flux.setup(Adam(learning_rate), model)

    @info "Start Training with params"
    @info "no. epochs: $nepochs"
    @info "no. batches: $nbatches"
    @info "learning rate: $learning_rate"
    @info "weight regularization: $weight_reg"

    pr = Progress(nepochs, desc="Training Progress", showspeed=true)

    for epoch in 1:nepochs
        loss = train_epoch!(model, settings, dataloader, opt_state)
        train_loss = compute_loss(model, settings, train_data)
        test_loss = compute_loss(model, settings, test_data)
        push!(acc_losses, loss)
        push!(train_losses, train_loss)
        push!(test_losses, test_loss)

        next!(pr;
            showvalues = [
                ("Epoch", epoch),
                ("Accumulated loss", loss),
                ("Train loss", train_loss),
                ("Test loss", test_loss)
            ]
        )

    end

    model = model |> cpu

    return model, acc_losses, train_losses, test_losses
    
end

"""
    plot_losses(train_losses, test_losses, settings::AbstractModelSettings; kwargs...)

Plot train and test losses

# Arguments

- `train_losses`: array containing train losses
- `test_losses`: array containing test losses
- `settings::AbstractModelSettings`: settings containing save dir

# Keywords

- `istart`: epoch number to start plotting from
    (**Default**: `1`)
"""
function plot_losses(train_losses, test_losses, settings::AbstractModelSettings; istart=1)
    plot(train_losses[istart:end], label="Train Loss", xlabel="Epoch", ylabel="Loss")
    plot!(test_losses[istart:end], label="Test Loss")
    savefig(joinpath(settings.model_dir, "train_test_losses.png"))
end

"""
    predict(model, settings::AbstractModelSettings, input_data)

Make a prediction using model and input_data

# Arguments

- `model`: trained model
- `settings::AbstractModelSettings`: settings containing hyperparameters, used for dispatch
- `input_data`: input data to base prediction on
"""
function predict(model, settings::AbstractModelSettings, input_data)
    error("Function predict not implemented for settings $(typeof(settings))")
end

"""
    plot_series(model, ts::TimeSeries, settings::AbstractModelSettings)

Compare and plot prediction from model with data in TimeSeries ts

# Arguments

- `model`: model used to make prediction
- `ts::TimeSeries`: time series with relevant input data and ground truth
- `settings::AbstractModelSettings`: settings containing hyperparameters, used for dispatch
"""
function plot_series(model, ts::TimeSeries, settings::AbstractModelSettings)
    error("Function plot_series not implemented for settings $(typeof(settings))")
end