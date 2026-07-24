# training_settings.jl
#
# Defines TrainingSettings — the hyperparameters that control the training loop
# and are not needed for model inference.
#
# All model-specific settings structs (TideSettings, SurgeSettings, etc.) have
# been stripped of these fields.  Training scripts construct both a model settings
# object and a TrainingSettings object; inference scripts only need the model one.

"""
    TrainingSettings

Hyperparameters that control the training loop and are not needed for inference.
All fields are shared across every model type.

# Fields

- `nepochs`: Number of training epochs.
    (**Default**: `100`)
- `batch_size`: Minibatch size for `Flux.DataLoader`. Each epoch iterates through
    all training data in shuffled batches of this size.
    (**Default**: `1024`)
- `learning_rate`: Initial learning rate for the Adam optimiser.
    (**Default**: `1.0e-3`)
- `lr_decay_factor`: Multiplicative decay applied to the learning rate every
    `lr_decay_epochs` epochs.  `nothing` disables decay.
    (**Default**: `nothing`)
- `lr_decay_epochs`: Epoch interval between learning-rate decay steps.
    (**Default**: `nothing`)
- `weight_decay`: L2 weight-decay coefficient. When `> 0`, Adam is wrapped in
    `OptimiserChain(WeightDecay(weight_decay), Adam)`.  `0.0` disables it.
    (**Default**: `0.0`)
- `early_stopping_epochs`: Stop training once the validation RMSE has not improved
    for this many consecutive epochs.  Requires validation data; `nothing`
    disables early stopping.
    (**Default**: `5`)
- `checkpoints`: Epoch numbers at which to save a model snapshot.  `nothing`
    disables checkpoints.
    (**Default**: `nothing`)
- `input_noise_std`: Standard deviation of Gaussian noise added to every model
    input tensor per batch during training (data-augmentation regularisation).
    `0.0` disables noise.
    (**Default**: `0.0`)
- `validation_split`: Fraction of the training data (taken from the end of the
    time series) held out as a validation set for loss reporting.  `0.0`
    disables validation.  An explicit `split = "validation"` data file takes
    priority over this fraction.
    (**Default**: `0.0`)
"""
@kwdef mutable struct TrainingSettings
    nepochs               = 100
    batch_size            = 1024
    learning_rate         = 1.0e-3
    lr_decay_factor       = nothing
    lr_decay_epochs       = nothing
    weight_decay          = 0.0
    early_stopping_epochs = 5
    checkpoints           = nothing
    input_noise_std       = 0.0
    validation_split      = 0.0
end

"""
    TrainingSettings(d::Dict) -> TrainingSettings

Construct a `TrainingSettings` from a plain dict (e.g. read from TOML).
Unknown keys are silently ignored; missing keys get their default values.
"""
function TrainingSettings(d::Dict)
    valid   = Set(fieldnames(TrainingSettings))
    unknown = setdiff(Set(Symbol(k) for k in keys(d)), valid)
    isempty(unknown) || error(
        "TrainingSettings: unknown train_settings key(s): " *
        join(sort!(String.(collect(unknown))), ", ") *
        ". Valid keys: " * join(sort!(String.(collect(valid))), ", ") * ".")
    kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d if Symbol(k) ∈ valid)
    return TrainingSettings(; kwargs...)
end

"""
    to_dict(ts::TrainingSettings) -> Dict{String,Any}

Convert a `TrainingSettings` to a plain dict suitable for TOML serialisation.
Fields set to `nothing` are omitted.
"""
function to_dict(ts::TrainingSettings)
    d = Dict{String,Any}()
    for f in fieldnames(TrainingSettings)
        v = getfield(ts, f)
        v !== nothing && (d[String(f)] = v)
    end
    return d
end
