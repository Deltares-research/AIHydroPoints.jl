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

function save_settings(settings::AbstractModelSettings)
    fn = joinpath(settings.model_dir, "settings.toml")
    # For convenience when loading we also write the type of the settings written
    tmp = Dict(key => getfield(settings, key) for key in propertynames(settings))
    dict = Dict(string(typeof(settings)) => tmp)
    open(fn, "w") do io
        TOML.print(io, dict)
    end
end

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

function save_model(model, settings::AbstractModelSettings)
    fn = joinpath(settings.model_dir, "model.jld2")
    jldsave(fn, model_state=Flux.state(model))
end

function load_model(settings::AbstractModelSettings, model_constructor)
    model = model_constructor(settings.model_pars)
    model_state = JLD2.load(joinpath(settings.model_dir, "model.jld2"), "model_state")
    Flux.loadmodel!(model, model_state)

    return model
end

function load_run(fn_dir, model_constructor)
    settings = load_settings(joinpath(fn_dir, "settings.toml"))
    model = load_model(settings, model_constructor)

    return model, settings
end

function prepare_train_data(ts::TimeSeries, settings::AbstractModelSettings)
    error("Function prepare_data not defined for settings $(typeof(settings))")
end

function prepare_inputs(data, settings::AbstractModelSettings)
    error("Function prepare_inputs not defined for settings $(typeof(settings))")    
end

function compute_loss(model, settings::AbstractModelSettings, data)
    error("Function compute loss not implemented for settings $(typeof(settings))")
end

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

function plot_losses(train_losses, test_losses, settings::AbstractModelSettings; istart=1)
    plot(train_losses[istart:end], label="Train Loss", xlabel="Epoch", ylabel="Loss")
    plot!(test_losses[istart:end], label="Test Loss")
    savefig(joinpath(settings.model_dir, "train_test_losses.png"))
end

function predict(model, settings::AbstractModelSettings, input_data)
    error("Function predict not implemented for settings $(typeof(settings))")
end

# function predict(model, settings::AbstractModelSettings, input_data)
#     inputs = prepare_inputs(settings, input_data...)
#     y_hat = model(inputs...)
#     return y_hat[:]
# end

function plot_series(model, ts::TimeSeries, settings::AbstractModelSettings; itimes=nothing)
    times = get_times(ts)
    station_names = get_names(ts)
    waterlevel = get_values(ts)
    nstations = length(station_names)

    if isnothing(itimes)
        itimes = 1:length(times)
    end
    times = times[itimes]

    for istation=1:nstations
        station_name = station_names[istation]
        prediction = predict(model, settings, (istation, times))
    end
end