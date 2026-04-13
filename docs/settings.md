# Settings Reference

Model settings are split into two distinct structs:

- **Model settings** (`TideSettings`, `SurgeSettings`, `WaveSettings`, `InteractionSettings`) —
  fields needed to construct the model and run inference.  These are saved and loaded with
  every trained model so that the model can be reconstructed without any training infrastructure.

- **`TrainingSettings`** — fields that control the training loop only.  They are not needed
  for inference.

Both structs are serialised together into a single `settings.toml` file under separate TOML
sections (e.g. `[TideSettings]` and `[TrainingSettings]`).

---

## TrainingSettings

Shared across all model types.

| Field | Type | Default | Description |
|---|---|---|---|
| `nepochs` | `Int` | `100` | Number of training epochs. |
| `nbatches` | `Int` | `1024` | Batch size passed to `Flux.DataLoader`. |
| `learning_rate` | `Float64` | `1.0e-3` | Initial learning rate for the Adam optimiser. |
| `lr_decay_factor` | `Float64` or `nothing` | `nothing` | Multiplicative LR decay factor; `nothing` disables decay. |
| `lr_decay_rate` | `Int` or `nothing` | `nothing` | Epoch interval between LR decay steps; `nothing` disables decay. |
| `weight_reg` | `Float64` | `1.0e-4` | L2 weight-decay coefficient (`WeightDecay` optimiser wrapper). |
| `patience` | `Int` | `5` | Epochs without improvement before early stopping. |
| `checkpoints` | `Vector{Int}` or `nothing` | `nothing` | Epoch numbers at which to save a model snapshot and diagnostic plots. |
| `val_daterange` | `Vector{String}` or `nothing` | `nothing` | Two ISO-8601 datetime strings defining the short validation window used for checkpoint plots. |
| `input_noise_std` | `Float64` | `0.0` | Std of Gaussian noise added to inputs during training (data-augmentation). Set `> 0` to enable. |

---

## TideSettings

Inference-time parameters for the tide model.

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `String` | `"MyTideModel"` | Name of the model (used for file naming). |
| `model_dir` | `String` | `"MyTideModel"` | Directory for saved model, plots, and settings. |
| `use_gpu` | `Bool` | `false` | Whether to use GPU for training and inference. |
| `nstations` | `Int` or `nothing` | `nothing` | Number of waterlevel stations; set from training data. |
| `freqs` | `Vector{String}` | `["SSA","K1","O1","Q1","P1","M2","S2","N2","K2","H"]` | Named tidal constituents. |
| `model_pars` | `Dict` | see below | Architecture parameters passed to `TideModel`. |

Default `model_pars` keys for `TideModel`:

| Key | Default | Description |
|---|---|---|
| `nlayers_branch` | `2` | Hidden layers in the branch network. |
| `nhidden_branch` | `64` | Width of the branch network. |
| `nlayers_trunk` | `0` | Hidden layers in the trunk network. |
| `nhidden_trunk` | `8` | Width of the trunk network. |
| `nlayers_down` | `1` | Hidden layers in the downsampling head. |

---

## SurgeSettings

Inference-time parameters for the surge model.

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `String` | `"MySurgeModel"` | Name of the model. |
| `model_dir` | `String` | `"MySurgeModel"` | Directory for saved model, plots, and settings. |
| `use_gpu` | `Bool` | `false` | Whether to use GPU. |
| `nstations` | `Int` or `nothing` | `nothing` | Number of surge stations; set from training data. |
| `nwind` | `Int` or `nothing` | `nothing` | Number of wind input stations; set from training data. |
| `nlags` | `Int` | `16` | Number of previous timesteps used as input. |
| `model_pars` | `Dict` | see below | Architecture parameters passed to `SurgeModel`. |

Default `model_pars` keys for `SurgeModel`:

| Key | Default | Description |
|---|---|---|
| `theta` | `10000.0` | Positional embedding frequency scale. |
| `nheads` | `4` | Number of attention heads. |
| `nlayers_branch` | `2` | Number of transformer layers. |
| `nlayers_trunk` | `0` | Hidden layers in the trunk network. |
| `nhidden_trunk` | `16` | Width of the trunk network. |
| `nembed` | `16` | Embedding dimension. |

---

## WaveSettings

Inference-time parameters for the wave model.

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `String` | `"MyWaveModel"` | Name of the model. |
| `model_dir` | `String` | `"MyWaveModel"` | Directory for saved model, plots, and settings. |
| `use_gpu` | `Bool` | `false` | Whether to use GPU. |
| `nstations` | `Int` or `nothing` | `nothing` | Number of wave output stations; set from training data. |
| `nwind` | `Int` or `nothing` | `nothing` | Number of wind input stations; set from training data. |
| `nlags` | `Int` | `16` | Number of previous timesteps used as input (must equal `2^length(nchannel)`). |
| `n_input_channels` | `Int` | `64` | Number of channels in the first convolutional layer. |
| `wind_scale` | `Float64` | `0.5` | Divisor applied to wind stress values at input. |
| `wave_scale` | `Float64` | `3.0` | Divisor applied to wave height targets. |
| `model_pars` | `Dict` | see below | Architecture parameters passed to `create_wave_model`. |

Default `model_pars` keys:

| Key | Default | Description |
|---|---|---|
| `nchannel` | `[64, 64, 64, 1]` | Output channels for each `Conv` layer. Length determines depth; `nlags` must equal `2^length`. |
| `activation` | `"swish"` | Activation function name (`"swish"` or `"relu"`). |

---

## TOML file format

Settings are stored in a two-section TOML file.  Example for the tide model:

```toml
[TideSettings]
model_name = "MyTideModel"
model_dir  = "models/MyTideModel"
use_gpu    = true
nstations  = 5
freqs      = ["SSA", "K1", "O1", "Q1", "P1", "M2", "S2", "N2", "K2", "H"]

    [TideSettings.model_pars]
    nlayers_branch = 2
    nhidden_branch = 64
    nlayers_trunk  = 0
    nhidden_trunk  = 8
    nlayers_down   = 1

[TrainingSettings]
nepochs         = 200
nbatches        = 1024
learning_rate   = 0.001
lr_decay_factor = 0.9
lr_decay_rate   = 50
weight_reg      = 0.0001
patience        = 10
val_daterange   = ["2011-01-01T00:00:00", "2011-01-15T00:00:00"]
checkpoints     = [40, 80, 120, 160]
```

Load and save:

```julia
# Loading
model_settings, train_settings = load_settings("path/to/settings.toml")

# Saving
save_settings(model_settings, train_settings)
```
