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
- `nbatches`: Batch size passed to `Flux.DataLoader`.
    (**Default**: `1024`)
- `learning_rate`: Initial learning rate for the Adam optimiser.
    (**Default**: `1.0e-3`)
- `lr_decay_factor`: Multiplicative decay applied to the learning rate every
    `lr_decay_rate` epochs.  `nothing` disables decay.
    (**Default**: `nothing`)
- `lr_decay_rate`: Epoch interval between learning-rate decay steps.
    (**Default**: `nothing`)
- `weight_reg`: L2 weight-decay coefficient (WeightDecay optimiser wrapper).
    (**Default**: `1.0e-4`)
- `patience`: Number of epochs without improvement before early stopping.
    (**Default**: `5`)
- `checkpoints`: Epoch numbers at which to save a model snapshot and diagnostic
    plots.  `nothing` disables checkpoints.
    (**Default**: `nothing`)
- `val_daterange`: Two-element vector of ISO-8601 datetime strings defining the
    short validation window used for checkpoint plots.
    (**Default**: `nothing`)
- `input_noise_std`: Standard deviation of Gaussian noise added to model inputs
    during training (data-augmentation regularisation).  `0.0` disables noise.
    Currently used by the wave model; set > 0 to enable for other models.
    (**Default**: `0.0`)
- `validation_split`: Fraction of the training data (taken from the end of the
    time series) held out as a validation set for loss reporting.  `0.0`
    disables validation.
    (**Default**: `0.0`)
"""
@kwdef mutable struct TrainingSettings
    nepochs          = 100
    nbatches         = 1024
    learning_rate    = 1.0e-3
    lr_decay_factor  = nothing
    lr_decay_rate    = nothing
    weight_reg       = 1.0e-4
    patience         = 5
    checkpoints      = nothing
    val_daterange    = nothing
    input_noise_std  = 0.0
    validation_split = 0.0
end
