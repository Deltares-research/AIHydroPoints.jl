# Training settings

Shared across all model types.  These fields live in `[train_settings]` in the TOML and are
not saved with the model — they are only needed during training.

| Field | Type | Default | Description |
|---|---|---|---|
| `nepochs` | `Int` | `100` | Number of training epochs. |
| `nbatches` | `Int` | `1024` | Minibatch size. |
| `learning_rate` | `Float64` | `1.0e-3` | Initial learning rate for the Adam optimiser. |
| `lr_decay_factor` | `Float64` or `nothing` | `nothing` | Multiplicative LR decay factor; `nothing` disables decay. |
| `lr_decay_rate` | `Int` or `nothing` | `nothing` | Epoch interval between LR decay steps; `nothing` disables decay. |
| `weight_reg` | `Float64` | `1.0e-4` | L2 weight-decay coefficient. |
| `patience` | `Int` | `5` | Epochs without improvement before early stopping. |
| `checkpoints` | `Vector{Int}` or `nothing` | `nothing` | Epoch numbers at which to save a model snapshot. |
| `validation_split` | `Float64` | `0.0` | Fraction of time series held out for validation (0 = disabled). |
| `val_daterange` | `Vector{String}` or `nothing` | `nothing` | Two ISO-8601 strings defining the short validation window used for checkpoint plots. |
| `input_noise_std` | `Float64` | `0.0` | Std of Gaussian noise added to inputs during training. |
