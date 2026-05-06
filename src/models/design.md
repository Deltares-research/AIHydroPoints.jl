
# Software design

This file describes the intended design for the model code, with the goal of
creating a more modular and reusable codebase. The main idea is to make models
object-like, with a common interface for all models, and to separate the
training settings from the model settings. This makes it easier to reuse the
model code for different training settings, and to run inference with a trained
model. We are mostly interested in ML models, but the design should be flexible
enough to accommodate other types of models as well.

The model type is generic: a model has input variables and output variables,
where each variable is a collection of time series for different stations.
Locations can but need not be the same for each variable. Computations are
causal: the output at a given time step only depends on the input at the
current and previous time steps.

The main datatypes are:
- `FooModel <: AbstractModel`: holds the model settings and parameters for a specific model. Concrete subtypes implement the common interface.
- `Dict{String, Any}` (model settings): holds the settings needed for constructing a model or for inference with a trained model.
- `TrainingSettings`: typed struct holding training hyperparameters (epochs, learning rate, etc.). Not needed for inference.

## AbstractModel (`abstract_model.jl`)

`AbstractModel` is the root supertype for all AI-Hydro forecast models.  A
concrete subtype is a self-contained object that stores its own settings and
trained parameters.

### Constructor convention

```julia
ConcreteModel(settings::Dict{String, Any}) -> ConcreteModel
```

### Required interface

| Method | Signature | Purpose |
|---|---|---|
| `predict`      | `(m::M, input::Dict{String,TimeSeries}) -> Dict{String,TimeSeries}` | Unified inference entry point |
| `get_settings` | `(m::M) -> Dict{String,Any}` | Return inference-time settings |
| `save_params`  | `(m::M, file::String; overwrite=false)` | Serialise trained weights to file |
| `load_params!` | `(m::M, file::String)` | Load weights from file in-place |
| `train_model!` | `(m::M, train_settings::TrainingSettings, input, target)` | Train in-place |

`TimeSeries` is the concrete type from `MultiTimeSeries.jl`.
All methods have fallback implementations that throw a descriptive error when a
subtype forgets to implement them.

`save_params` checks that the parent directory exists and raises an error if the
file already exists and `overwrite=false` (default).

## AbstractFluxModel (`abstract_flux_model.jl`)

`AbstractFluxModel <: AbstractModel` sits between `AbstractModel` and concrete
Flux-based model types.  It implements `predict`, `save_params`, and
`load_params!` once, delegating to three customisation points and `get_flux_model`.

```
AbstractModel
    └── AbstractFluxModel   — implements predict, save_params, load_params!
            └── FooModel    — domain-specific concrete subtype
```

### predict pipeline

```julia
function predict(model::AbstractFluxModel, input::Dict{String, TimeSeries})
    tensor, output = preprocess(model, input)   # build tensor + pre-allocate output
    y = forward(model, tensor)                  # Flux forward pass
    postprocess!(output, model, y)              # fill output in-place
    return output
end
```

`preprocess` returns both the input tensor and a pre-allocated
`Dict{String, TimeSeries}` whose `values` matrices are zero-initialised with
the correct shape and metadata (times, station names, coordinates).
`postprocess!` fills those matrices in-place — this avoids storing metadata as
a side-effect on the model struct and keeps `TimeSeries` allocation in one place.

### Tensor layout

| Tensor | Shape | Notes |
|---|---|---|
| Input (from `preprocess`) | `(locations, features, time_lag, time)` | column-major; `time` is the batch dim |
| Output (from `forward`)   | `(locations, features, time)` | one output per time step |

`time_lag = 1` means no lag (single time step input).

### Required customisation points

| Method | Signature | Purpose |
|---|---|---|
| `preprocess`    | `(m::M, input::Dict{String,TimeSeries}) -> (Array{Float32,4}, Dict{String,TimeSeries})` | Build input tensor; pre-allocate output |
| `forward`       | `(m::M, x::Array{Float32,4}) -> Array{Float32,3}` | Reshape + Flux forward pass |
| `postprocess!`  | `(output::Dict{String,TimeSeries}, m::M, y::Array{Float32,3})` | Fill output values in-place |
| `get_flux_model`| `(m::M) -> <Flux model>` | Expose chain for save/load |
| `get_settings`  | `(m::M) -> Dict{String,Any}` | Return inference-time settings |

`save_params` and `load_params!` are implemented once at this level using
`get_flux_model` together with `Flux.state` / `Flux.loadmodel!` and JLD2.

## AbstractSurgeModel (`AbstractSurgeModel.jl`)

`AbstractSurgeModel <: AbstractFluxModel` is an intermediate abstract type that
captures shared logic for all surge models.  Concrete subtypes only need to
implement `forward` and `get_flux_model` / `get_settings`.

### Shared implementations

| Method | What it does |
|---|---|
| `preprocess` | Builds `(1, 3*nwind, nlags, nvalid)` wind/pressure lag tensor and pre-allocates output |
| `postprocess!` | Writes `y[:, 1, :]` into `output["surge"].values` |
| `train_model!` | Adam loop with ProgressMeter, temporal train/val split, returns `(train_losses, val_losses)` |

### Input key handling

`preprocess` accepts either `"stress_x"`/`"stress_y"` (used directly) or
`"wind_x"`/`"wind_y"` (converted via `uv_to_stress_xy`).  The helper
`_get_stress(input)` encapsulates this.

```
AbstractModel
    └── AbstractFluxModel   — predict, save_params, load_params!
            └── AbstractSurgeModel  — preprocess, postprocess!, train_model!
                    ├── LinearSurgeModel
                    └── AttentionSurgeModel
```

## LinearSurgeModel (`LinearSurgeModel.jl`)

`LinearSurgeModel <: AbstractSurgeModel` is the simplest concrete implementation,
using a single `Dense` layer (identity activation — linear regression) to
predict storm surge from wind-stress and pressure history.

### Settings

```julia
settings = Dict{String, Any}(
    "nstations" => 5,   # number of output (waterlevel) stations
    "nwind"     => 9,   # number of input (forcing) stations
    "nlags"     => 16,  # number of previous time steps used as input
)
model = LinearSurgeModel(settings)
```

### Input dict keys

| Key | Shape | Description |
|---|---|---|
| `"stress_x"` or `"wind_x"` | `(nwind, T)` | East wind-stress (or velocity, auto-converted) |
| `"stress_y"` or `"wind_y"` | `(nwind, T)` | North wind-stress (or velocity, auto-converted) |
| `"pressure"` | `(nwind, T)` | Sea-level pressure (scaled: `2e-4*(p - 1e5)`) |

### Data flow

```
preprocess → tensor (1, 3*nwind, nlags, ntimes_valid)
                + output Dict("surge" => zeros TimeSeries)
forward    → flatten → Dense(3*nwind*nlags => nstations) → (nstations, 1, ntimes_valid)
postprocess! → output["surge"].values .= y[:, 1, :]
```

### train_model!

```julia
train_losses, val_losses = train_model!(model, train_settings, input, target)
```

`input` and `target` are `Dict{String,TimeSeries}`.  If
`train_settings.validation_split > 0`, the last fraction of the time series is
held out for validation and `val_losses` is populated; otherwise it is empty.
Training progress is shown via a `ProgressMeter` bar with per-epoch RMSE, and
`@info` lines are emitted every `nepochs ÷ 10` epochs.

## AttentionSurgeModel (`AttentionSurgeModel.jl`)

`AttentionSurgeModel <: AbstractSurgeModel` uses a transformer branch network
for wind/pressure history and a dense trunk network for station metadata, merged
via graph-adjacency weights.

### Constructor

```julia
model = AttentionSurgeModel(settings::Dict{String, Any}, gn::GraphNetwork)
```

Required keys in `settings`: `"nstations"`, `"nwind"`, `"nlags"`, `"model_pars"`.

Required keys in `"model_pars"`:

| Key | Description |
|---|---|
| `"nembed"` | Embedding dimension |
| `"theta"` | RoPE/SinCos frequency base |
| `"nheads"` | Transformer attention heads |
| `"nlayers_branch"` | Transformer layers in branch network |
| `"nlayers_trunk"` | Dense layers in trunk network |
| `"nhidden_trunk"` | Hidden width of trunk network |

### Overridden methods

`AttentionSurgeModel` overrides `preprocess`, `forward`, and `train_model!`
(inherited defaults from `AbstractSurgeModel` do not apply because the model
has two inputs):

| Method | Notes |
|---|---|
| `preprocess` | Returns `((x_station, x_wind), output)` — dual input |
| `forward` | Accepts `Tuple`, takes `[:, end, :]` of the output |
| `train_model!` | Batches over both `x_station` and `x_wind` simultaneously |

### Data flow

```
preprocess → x_wind    (3*nwind, nlags, ntimes_valid)
             x_station (6, nstations, ntimes_valid)   [cos/sin lat, lon, day-of-year]
forward    → AttentionSurgeFlux → (nstations, nlags, ntimes) → [:, end, :] → (nstations, 1, ntimes)
postprocess! → output["surge"].values .= y[:, 1, :]
```

## AbstractTideModel (`AbstractTideModel.jl`)

`AbstractTideModel <: AbstractFluxModel` is an intermediate abstract type that
captures shared logic for all tide models.  Tide models are astronomically
driven — no external forcing.  The only input is a `"waterlevel"` TimeSeries
that provides times and station coordinates; Doodson numbers are computed from
those automatically.

### Shared implementations

| Method | What it does |
|---|---|
| `preprocess` | Builds `(4, nstations, ntimes)` station tensor and `(2*nfreqs, ntimes)` Doodson tensor; pre-allocates output |
| `postprocess!` | Writes `y[:, 1, :]` into `output["waterlevel"].values` |
| `train_model!` | Adam loop with ProgressMeter, temporal train/val split, returns `(train_losses, val_losses)` |

### Input convention

Both `input` and `target` carry a `"waterlevel"` key.  At training time they
point to the same `TimeSeries`.

```
AbstractModel
    └── AbstractFluxModel   — predict, save_params, load_params!
            └── AbstractTideModel  — preprocess, postprocess!, train_model!
                    └── DeepONetTideModel
```

## DeepONetTideModel (`DeepONetTideModel.jl`)

`DeepONetTideModel <: AbstractTideModel` wraps the `TideModel` Flux
architecture from `src/tides.jl` (branch network for Doodson arguments, trunk
network for station coordinates, merged and downsampled per station).

### Constructor

```julia
model = DeepONetTideModel(settings::Dict{String, Any})
```

Required keys in `settings`:

| Key | Description |
|---|---|
| `"freqs"` | Vector of tidal constituent names, e.g. `["M2","S2","K1",...]` |
| `"model_pars"` | Dict with `"nlayers_branch"`, `"nhidden_branch"`, `"nlayers_trunk"`, `"nhidden_trunk"`, `"nlayers_down"` |

### Data flow

```
preprocess → x_station (4, nstations, ntimes)   [cos/sin lat, cos/sin lon]
             x_doodson (2*nfreqs, ntimes)        [cos/sin Doodson arguments]
forward    → TideModel(x_station, x_doodson) → (nstations, ntimes) → (nstations, 1, ntimes)
postprocess! → output["waterlevel"].values .= y[:, 1, :]
```

## Utilities

### `toml_utils.jl` — `toml_write`

```julia
toml_write(path::String, dict::Dict; overwrite::Bool=false)
```

Writes a `Dict{String,Any}` to a TOML file. Raises an error if the parent
directory does not exist or if the file exists and `overwrite=false`.
Use this to persist model settings alongside weights:

```julia
toml_write(joinpath(save_dir, "settings.toml"), get_settings(model); overwrite=true)
```

### `plot_utils.jl` — `save_loss_plot`

```julia
save_loss_plot(path::String, train_losses::Vector, val_losses::Vector=[]; overwrite::Bool=false)
```

Saves a PNG plot of train (and optionally val) RMSE against epoch. Same
directory/overwrite guards as `toml_write` and `save_params`.

## plot_series (`plot_utils.jl` + model files)

`plot_series` is part of the `AbstractModel` interface.  Each intermediate
abstract type provides its own implementation; the shared plotting skeleton
lives in `_plot_station_series` in `src/plot_utils.jl`.

```julia
plot_series(model, input::Dict{String,TimeSeries}, target::Dict{String,TimeSeries},
            series_name::String; save_dir, timerange, station_names, show_fft)
```

`_plot_station_series` aligns target to prediction times (handles lag trimming
in surge models via `select_timespan`), computes per-station RMSE, and saves
one PNG per station.  It uses `Plots.plot(ts; location_index=i)` from
`MultiTimeSeries.jl` for the observation panel.

| Model type | `show_fft` | Layout |
|---|---|---|
| `AbstractSurgeModel` | not supported | 2-panel (series + residual) |
| `AbstractTideModel`  | optional      | 2-panel or 4-panel (+ FFT panels) |

## Notes

- The new model hierarchy (`AbstractFluxModel` and subtypes) coexists with the
  old concrete models (`TideSettings`, `SurgeSettings`, `WaveSettings`) in
  `training.jl`.  The old models will be migrated incrementally.
