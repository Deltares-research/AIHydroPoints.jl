# training.jl
# Functions to do training and saving and loading training runs
# Each new model type implements a settings type (subtype of AbstractModelSettings)
# and uses it to dispatch the various functions defined here.

using TOML
using JLD2
using Plots
using Statistics
using ProgressMeter: Progress, next!
using ParameterSchedulers: Constant, Step

"""
    AbstractModelSettings

Base type for inference-time model settings structs (`TideSettings`, `SurgeSettings`, etc.).
Each subtype holds only the fields needed to construct the model and run inference:
`model_name`, `model_dir`, `use_gpu`, `nstations`, and model-specific architecture fields.

Training hyperparameters (epochs, learning rate, etc.) are held in a separate `TrainingSettings`
struct so that inference scripts do not need to carry them.
"""
abstract type AbstractModelSettings end

"""
    save_settings(model_settings::AbstractModelSettings, train_settings::TrainingSettings)

Save model and training settings to `"settings.toml"` inside `model_settings.model_dir`.
The file contains two top-level TOML sections: one named after the concrete model settings
type (e.g. `[TideSettings]`) and one `[TrainingSettings]`.

# Arguments

- `model_settings`: Inference-time model settings.
- `train_settings`: Training hyperparameters.
"""
function save_settings(model_settings::AbstractModelSettings, train_settings::TrainingSettings)
    fn = joinpath(model_settings.model_dir, "settings.toml")
    model_dict = Dict(string(key) => getfield(model_settings, key)
                      for key in propertynames(model_settings)
                      if !isnothing(getfield(model_settings, key)))
    train_dict = Dict(string(key) => getfield(train_settings, key)
                      for key in propertynames(train_settings)
                      if !isnothing(getfield(train_settings, key)))
    dict = Dict(
        string(typeof(model_settings)) => model_dict,
        "TrainingSettings"             => train_dict,
    )
    open(fn, "w") do io
        TOML.print(io, dict)
    end
end

"""
    load_settings(fn) -> (model_settings, train_settings)

Load settings from a TOML file previously written by `save_settings`.

Returns a tuple `(model_settings, train_settings)` where `model_settings` is an
instance of the concrete model type recorded in the file (e.g. `TideSettings`) and
`train_settings` is a `TrainingSettings`.  If the file has no `[TrainingSettings]`
section, `train_settings` is constructed with all defaults.

# Arguments

- `fn`: Path to the `settings.toml` file.
"""
function load_settings(fn)
    dict = TOML.parsefile(fn)
    model_key = only(filter(k -> k != "TrainingSettings", keys(dict)))
    sett_type = @eval $(Symbol(model_key))
    vals = dict[model_key]
    model_settings = sett_type(; (Symbol.(keys(vals)) .=> values(vals))...)

    train_vals = get(dict, "TrainingSettings", Dict{String,Any}())
    train_settings = TrainingSettings(; (Symbol.(keys(train_vals)) .=> values(train_vals))...)

    return model_settings, train_settings
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

Load a training run from directory.  Returns the reconstructed model and the inference-time
model settings (training settings are discarded).

# Arguments

- `fn_dir`: Path to run directory.
- `model_constructor`: Function that builds the model; used to create a blank model into
  which saved parameters are loaded.
"""
function load_run(fn_dir, model_constructor)
    model_settings, _ = load_settings(joinpath(fn_dir, "settings.toml"))
    model = load_model(model_settings, model_constructor)
    return model, model_settings
end

"""
    prepare_train_data(ts::TimeSeries, settings::AbstractModelSettings)

Create training data from a TimeSeries ts using the hyperparameters in settings.

# Arguments

- `ts::TimeSeries`: Input time series
- `settings::AbstractModelSettings`: settings containing hyperparameters
"""
function prepare_train_data(data_dict::Dict{String, <:AbstractTimeSeries}, settings::AbstractModelSettings)
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
    train_epoch!(model, settings::AbstractModelSettings, train_settings::TrainingSettings, dataloader, opt_state)

Train the model for one epoch.

# Arguments

- `model`: Model to train.
- `settings`: Inference-time model settings; used for dispatch.
- `train_settings`: Training hyperparameters (e.g. `input_noise_std`).
- `dataloader`: `Flux.DataLoader` with training data.
- `opt_state`: Optimiser state.
"""
function train_epoch!(model, settings::AbstractModelSettings, train_settings::TrainingSettings, dataloader, opt_state)
    error("Function train_epoch! not implemented for settings $(typeof(settings))")
end

"""
    train_model(model, model_settings::AbstractModelSettings, train_settings::TrainingSettings, train_dict, test_dict)

Train `model` using `train_dict`, evaluate on `test_dict`.

Model-specific dispatch (data preparation, loss, forward pass) is driven by `model_settings`.
All training hyperparameters are read from `train_settings`.

# Arguments

- `model`: Model to train.
- `model_settings`: Inference-time settings; controls dispatch of `prepare_train_data`,
  `compute_loss`, `train_epoch!`, `plot_series`.
- `train_settings`: Training hyperparameters (epochs, batch size, learning rate, etc.).
- `train_dict`: Dict of `AbstractTimeSeries` used for training.
- `test_dict`: Dict of `AbstractTimeSeries` used for evaluation.
"""
function train_model(model, model_settings::AbstractModelSettings, train_settings::TrainingSettings,
                     train_dict::Dict{String, <:AbstractTimeSeries}, test_dict::Dict{String, <:AbstractTimeSeries})
    nepochs         = train_settings.nepochs
    nbatches        = train_settings.nbatches
    learning_rate   = train_settings.learning_rate
    lr_decay_factor = train_settings.lr_decay_factor
    lr_decay_rate   = train_settings.lr_decay_rate
    weight_reg      = train_settings.weight_reg
    patience        = train_settings.patience
    checkpoints     = train_settings.checkpoints
    val_daterange   = train_settings.val_daterange
    use_gpu         = model_settings.use_gpu

    lr_schedule = Constant(learning_rate)
    if !isnothing(lr_decay_factor) && !isnothing(lr_decay_rate)
        lr_schedule = Step(start=learning_rate, decay=lr_decay_factor, step_sizes=lr_decay_rate)
    end

    if use_gpu && CUDA.has_cuda()
        @info "Training on GPU"
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    train_losses = []
    test_losses  = []
    acc_losses   = []

    tmp_losses = 1e3 * ones(patience)

    train_data = prepare_train_data(train_dict, model_settings)
    test_data  = prepare_train_data(test_dict,  model_settings)

    model      = model      |> device
    train_data = train_data |> device
    test_data  = test_data  |> device
    dataloader = Flux.DataLoader(train_data, batchsize=nbatches, shuffle=true)

    opt_state = Flux.setup(OptimiserChain(WeightDecay(weight_reg), Adam(learning_rate)), model)

    @info "Start Training with params"
    @info "no. epochs: $nepochs"
    @info "no. batches: $nbatches"
    @info "learning rate: $learning_rate"
    @info "weight regularization: $weight_reg"

    pr = Progress(nepochs, desc="Training Progress", showspeed=true)

    for epoch in 1:nepochs
        loss       = train_epoch!(model, model_settings, train_settings, dataloader, opt_state)
        train_loss = compute_loss(model, model_settings, train_data)
        test_loss  = compute_loss(model, model_settings, test_data)
        push!(acc_losses,   loss)
        push!(train_losses, train_loss)
        push!(test_losses,  test_loss)

        next!(pr;
            showvalues = [
                ("Epoch",            epoch),
                ("Accumulated loss", loss),
                ("Train loss",       train_loss),
                ("Test loss",        test_loss),
                ("Learning rate",    lr_schedule(epoch))
            ]
        )

        if test_loss <= mean(tmp_losses)
            push!(tmp_losses, test_loss)
            popfirst!(tmp_losses)
        else
            @info "No improvement in test loss for $patience epochs, stopping training"
            break
        end

        if !isnothing(checkpoints) && epoch in checkpoints
            @info "Creating checkpoint $epoch"
            tmp_settings = deepcopy(model_settings)
            tmp_settings.model_dir = joinpath(model_settings.model_dir, "checkpoints", "checkpoint_$epoch")
            if !isdir(tmp_settings.model_dir)
                mkpath(tmp_settings.model_dir)
            end
            save_model(model |> cpu, tmp_settings)

            valdays = day.(DateTime.(val_daterange))
            indx    = 24 * ((valdays[2] - valdays[1]) + 1)
            train_daterange = [get_times(train_dict["waterlevel"])[1],
                               get_times(train_dict["waterlevel"])[indx]]
            plot_series(model |> cpu, tmp_settings, test_dict,  "chk_$(epoch)_test";  write_series=false)
            plot_series(model |> cpu, tmp_settings, test_dict,  "chk_$(epoch)_test_short";  write_series=false, timerange=val_daterange)
            plot_series(model |> cpu, tmp_settings, train_dict, "chk_$(epoch)_train"; write_series=false)
            plot_series(model |> cpu, tmp_settings, train_dict, "chk_$(epoch)_train_short"; write_series=false, timerange=train_daterange)
        end

        if !isnothing(lr_schedule)
            Optimisers.adjust!(opt_state, eta=lr_schedule(epoch + 1))
        end
    end

    model = model |> cpu

    return model, acc_losses, train_losses, test_losses
end

"""
    plot_losses(train_losses, test_losses, model_settings, train_settings; kwargs...)

Plot train and test losses and save to `model_settings.model_dir`.

# Arguments

- `train_losses`: Array of per-epoch training losses.
- `test_losses`: Array of per-epoch test losses.
- `model_settings`: Model settings; used only for the output directory.
- `train_settings`: Training settings; used for checkpoints and LR schedule.

# Keywords

- `istart`: First epoch to include in the plot.
    (**Default**: `1`)
"""
function plot_losses(train_losses, test_losses,
                     model_settings::AbstractModelSettings, train_settings::TrainingSettings;
                     istart=1)
    plot(train_losses[istart:end], label="Train Loss", xlabel="Epoch", ylabel="Loss", minorgrid=true)
    plot!(test_losses[istart:end], label="Test Loss")
    if !isnothing(train_settings.checkpoints)
        vline!(train_settings.checkpoints, label="Checkpoints", ls=:dot, lc=:red)
    end
    if !isnothing(train_settings.lr_decay_factor) && !isnothing(train_settings.lr_decay_rate)
        schedule = Step(start=train_settings.learning_rate, decay=train_settings.lr_decay_factor,
                        step_sizes=train_settings.lr_decay_rate)
        y_lims = (floor(log10(schedule(train_settings.nepochs))), ceil(log10(schedule(1))))
        plot!(twinx(), 1:train_settings.nepochs, log10.(schedule.(1:train_settings.nepochs)),
              label=false, lc=:black, ylims=y_lims, yaxis="Learning Rate (log10)")
    end
    xlims!(istart, length(train_losses) + istart)
    plot!(legend=:topright)
    savefig(joinpath(model_settings.model_dir, "train_test_losses.png"))
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
function plot_series(model, data_dict::Dict{String, <:AbstractTimeSeries}, settings::AbstractModelSettings, series_name)
    error("Function plot_series not implemented for settings $(typeof(settings))")
end