# Model settings

Model settings are stored as a plain `Dict{String,Any}` and saved alongside every trained
model as `model_settings.toml` so the model can be reconstructed without any training
infrastructure.  For inference, `[model_settings]` only requires `model_dir`.

## Surge model settings

The surge model takes lagged wind stress and pressure at `nlocations_input` forcing locations
and predicts storm surge at `nlocations_output` output stations.

Concrete models: `LinearSurgeModel`, `ConvSurgeModel`, `AttentionSurgeModel`.

### Required keys at construction

| Key | Description |
|---|---|
| `"model_name"` | Model type; must match a key in `MODEL_REGISTRY` (e.g. `"ConvSurgeModel"`). |
| `"nlocations_output"` | Number of output (surge) stations. |
| `"nlocations_input"` | Number of input (wind/pressure) locations. |
| `"nlags"` | Number of lagged time steps used as input. |

### Optional keys (with defaults)

| Key | Default | Description |
|---|---|---|
| `"model_dir"` | derived from `runid` + `model_name` | Directory for saved model files. |
| `"use_gpu"` | `false` | Whether to use GPU. |
| `"model_pars"` | model-dependent | Architecture parameters (see below). |

### Auto-populated by `validate_and_augment_settings!`

`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantities"`, `"nlocations_output"`,
`"in_names"`, `"in_lons"`, `"in_lats"`, `"in_quantities"`, `"nlocations_input"`,
`"model_dir"`, `"model_weights"`.

### `model_pars` for `ConvSurgeModel`

| Key | Default | Description |
|---|---|---|
| `"channels"` | `[32, 16]` | Output channels for each Conv layer. |
| `"filtersize"` | `3` | Conv kernel width **and stride** (`stride = filtersize`): windows tile the lag axis with no overlap, so the lag length shrinks by `filtersize` per layer (`cld(·, filtersize)`), reducing the flattened `Dense` input. |
| `"activation"` | `"swish"` | Activation function (`"swish"` or `"relu"`). |

### `model_pars` for `AttentionSurgeModel`

| Key | Default | Description |
|---|---|---|
| `"nembed"` | `32` | Embedding dimension. |
| `"theta"` | `1000.0` | Positional embedding frequency scale. |
| `"nheads"` | `4` | Number of attention heads. |
| `"nlayers_branch"` | `2` | Number of transformer layers. |
| `"nlayers_trunk"` | `2` | Hidden layers in the trunk network. |
| `"nhidden_trunk"` | `32` | Width of the trunk network. |

---

## Tide model settings

The tide model takes Doodson phases (computed from time) and station identity as input and
predicts tidal water level at `nstations` output stations.

Concrete models: `DeepONetTideModel`, `ProductTideModel`.

### Required keys at construction

| Key | Description |
|---|---|
| `"model_name"` | Model type; must match a key in `MODEL_REGISTRY` (e.g. `"DeepONetTideModel"`). |
| `"freqs"` | List of named tidal constituents (e.g. `["M2", "S2", "K1"]`). |

### Optional keys (with defaults)

| Key | Default | Description |
|---|---|---|
| `"model_dir"` | derived from `runid` + `model_name` | Directory for saved model files. |
| `"use_gpu"` | `false` | Whether to use GPU. |
| `"model_pars"` | model-dependent | Architecture parameters (see below). |

### Auto-populated by `validate_and_augment_settings!`

`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantities"`, `"nlocations_output"`,
`"model_dir"`, `"model_weights"`.
(Tide models have no loaded inputs so `in_*` keys are not set.)

### `model_pars` for `DeepONetTideModel`

| Key | Default | Description |
|---|---|---|
| `"nlayers_branch"` | `2` | Hidden layers in the Doodson branch network. |
| `"nhidden_branch"` | `64` | Width of the branch network. |
| `"nlayers_trunk"` | `0` | Hidden layers in the station trunk network. |
| `"nhidden_trunk"` | `8` | Width of the trunk network. |
| `"nlayers_down"` | `1` | Hidden layers in the downsampling head. |

---

## Wave model settings

The wave model takes lagged wind speed and direction at `nlocations_input` input locations
and predicts significant wave height at `nlocations_output` output stations.  Station
identity is encoded via a one-hot vector so each (station × time) pair is an independent
sample.

Concrete models: `ConvWaveModel`, `DeepONetWaveModel`.

### Required keys at construction

| Key | Description |
|---|---|
| `"model_name"` | Model type; must match a key in `MODEL_REGISTRY` (e.g. `"ConvWaveModel"`). |
| `"nlocations_output"` | Number of output (wave height) stations. |
| `"nlocations_input"` | Number of input (wind) locations. |
| `"nlags"` | Number of lagged time steps; must equal `2^length(model_pars["nchannel"])`. |

### Optional keys (with defaults)

| Key | Default | Description |
|---|---|---|
| `"model_dir"` | derived from `runid` + `model_name` | Directory for saved model files. |
| `"use_gpu"` | `false` | Whether to use GPU. |
| `"wind_scale"` | `0.5` | Divisor applied to wind stress values at input. |
| `"wave_scale"` | `3.0` | Divisor applied to wave height targets during training. |
| `"n_input_channels"` | `64` | Channels in the first convolutional layer (`ConvWaveModel` only). |
| `"model_pars"` | model-dependent | Architecture parameters (see below). |

### Auto-populated by `validate_and_augment_settings!`

`"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantities"`, `"nlocations_output"`,
`"in_names"`, `"in_lons"`, `"in_lats"`, `"in_quantities"`, `"nlocations_input"`,
`"model_dir"`, `"model_weights"`.

### `model_pars` for `ConvWaveModel`

| Key | Default | Description |
|---|---|---|
| `"nchannel"` | `[64, 64, 64, 1]` | Output channels per Conv layer. Length determines depth; `nlags` must equal `2^length`. |
| `"activation"` | `"swish"` | Activation function (`"swish"` or `"relu"`). |

---

## Interaction model settings

The interaction model takes lagged tide and surge at each station and predicts the non-linear
tide–surge interaction (residual water level).  Station identity is encoded via a one-hot
vector.  Inputs and outputs are Z-score normalised; the normalisation statistics are computed
from training data and stored in the settings so inference uses identical scaling.

Concrete models: `ConvInteractionModel`.

### Required keys at construction

| Key | Description |
|---|---|
| `"model_name"` | Model type; must match a key in `MODEL_REGISTRY` (e.g. `"ConvInteractionModel"`). |
| `"nlocations_output"` | Number of output (interaction) stations. |
| `"nlags"` | Number of lagged time steps; must equal `2^length(model_pars["channels"])`. |

### Optional keys (with defaults)

| Key | Default | Description |
|---|---|---|
| `"model_dir"` | derived from `runid` + `model_name` | Directory for saved model files. |
| `"use_gpu"` | `false` | Whether to use GPU. |
| `"model_pars"` | `Dict("channels" => [32, 16, 1])` | Architecture parameters (see below). |

### Auto-populated by `validate_and_augment_settings!` and `train_model!`

`validate_and_augment_settings!`: `"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantities"`,
`"nlocations_output"`, `"in_names"`, `"in_lons"`, `"in_lats"`, `"in_quantities"`,
`"nlocations_input"`, `"model_dir"`, `"model_weights"`.

`train_model!`: `"input_mu"`, `"input_std"`, `"output_mu"`, `"output_std"` (Z-score statistics).

### `model_pars` for `ConvInteractionModel`

| Key | Default | Description |
|---|---|---|
| `"channels"` | `[32, 16, 1]` | Output channels per Conv layer (last must be 1). `nlags` must equal `2^length`. |

---

## TOML file format

After training, model settings are saved as a flat TOML file via `toml_write`.
A `[model_pars]` subsection holds the architecture dict.

Example for `ConvWaveModel`:

```toml
model_name         = "ConvWaveModel"
model_dir          = "models/ConvWaveModel"
nlocations_output  = 3
nlocations_input   = 3
nlags              = 16
wind_scale = 0.5
wave_scale = 3.0
n_input_channels = 64
out_names    = ["Europlatform", "F3", "K13a"]
out_quantity = "wave_height"
out_lons     = [3.275201, 4.716978, 3.219036]
out_lats     = [51.998767, 54.851299, 53.218117]

[model_pars]
nchannel   = [64, 64, 64, 1]
activation = "swish"
```

Files written by `train()` to `model_dir`:

| File | Contents |
|---|---|
| `model_settings.toml` | Full model settings dict including `model_weights` |
| `run_settings.toml` | Complete run config (all TOML tables) for reproducibility |
| `params.jld2` | Weights at the final training epoch |
| `params_best.jld2` | Weights at the best validation loss epoch (written when validation data is provided) |
| `params_epoch_N.jld2` | Epoch snapshots, for each `N` listed in `train_settings.checkpoints` |
| `losses.png` | Training and validation loss curves |

`model_weights` in `model_settings.toml` points to `params_best.jld2` when that file exists,
otherwise to `params.jld2`.  `predict()` loads from this file.  To use a specific checkpoint
for inference, set `model_weights = "params_epoch_50.jld2"` in the predict TOML.
