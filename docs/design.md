
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
    x, output = preprocess(model, input)   # build model-specific input + pre-allocate output
    y = forward(model, x)                  # Flux forward pass
    postprocess!(output, model, y)         # fill output in-place
    return output
end
```

`preprocess` returns both the input `x` and a pre-allocated
`Dict{String, TimeSeries}` whose `values` matrices are zero-initialised with
the correct shape and metadata (times, station names, coordinates).
`postprocess!` fills those matrices in-place — this avoids storing metadata as
a side-effect on the model struct and keeps `TimeSeries` allocation in one place.

### Tensor layout — per-model, not unified

There is **no single imposed tensor shape** at this level (see
`docs/notes_dimensions.md`, the tensor-layout refactor). Each model declares the layout its
Flux layers need; the only thing standardised here is the *container* exchanged
between the customisation points:

- `preprocess` returns `(x, output)` where `x` is a **tuple of tensors** (a
  1-tuple for single-input models, an N-tuple for multi-input models such as
  `AttentionSurgeModel`). Batch/sample is the **last** axis of every tensor, so
  `Flux.DataLoader((x, y))` batches them consistently (nested tuples are batched
  element-wise).
- `forward` returns a **2-D** array (`(locations, time)` for surge/tide, or
  `(1, nsamples)` for the station×time wave/interaction models).

**All four families** now follow this convention (since the package-wide
`train_model!` unification). The Flux model is
called uniformly as `get_flux_model(m)(x)` — every flux model is callable on its
input tuple (single-input Dense/Conv chains prepend `only` to unwrap the 1-tuple;
multi-arg models carry a `(m)(x::Tuple) = m(x...)` method). As a result `forward`
and `train_model!` are provided **generically** at this level and no family
overrides them; each per-family section below documents only its own
`preprocess`/`postprocess!` layout.

### Required customisation points

| Method | Signature | Purpose |
|---|---|---|
| `preprocess` (predict) | `(m::M, input) -> (Tuple, Dict{String,TimeSeries})` | Build model-specific input tuple; pre-allocate output |
| `preprocess` (train)   | `(m::M, input, target) -> (Tuple, Matrix)` | Build input tuple + target `y` in flux-output space; fit any train-time stats |
| `postprocess!`  | `(output::Dict{String,TimeSeries}, m::M, y)` | Fill output values in-place from 2-D `y` |
| `get_flux_model`| `(m::M) -> <Flux model>` | Expose chain for save/load; must be callable as `flux(x::Tuple)` |
| `get_settings`  | `(m::M) -> Dict{String,Any}` | Return inference-time settings |

`forward(::AbstractFluxModel, x) = get_flux_model(m)(x)` and
`train_model!(::AbstractFluxModel, …)` are **provided** here, not customisation
points — every family shares one training loop (`Flux.DataLoader` over the input
tuple, `m(x)` per batch, checkpointing, val split, lr-decay). The only per-family
training seam is the 2-arg vs 3-arg `preprocess`.

`save_params` and `load_params!` are implemented once at this level using
`get_flux_model` together with `Flux.state` / `Flux.loadmodel!` and JLD2.

## AbstractSurgeModel (`AbstractSurgeModel.jl`)

`AbstractSurgeModel <: AbstractFluxModel` is an intermediate abstract type that
captures the surge-specific data extraction.  Concrete subtypes implement
`preprocess` (their own tensor assembly) plus `get_flux_model` / `get_settings`;
`forward` and `train_model!` are inherited from `AbstractFluxModel` (generic for
all families), and `postprocess!` + the train-form `preprocess(m, input, target)`
are shared at this surge level.

### Shared implementations

| Method | What it does |
|---|---|
| `_surge_lag_windows` | Shared **data extraction**: aligns locations, converts wind→stress, scales pressure, and slices each field into `(nwind, nlags, nvalid)` lag windows. Each model's `preprocess` calls this, then assembles the windows into its own layout. |
| `_alloc_surge_output` | Allocates the zero-initialised `Dict("surge" => ts)` output container. |
| `preprocess(m, input, target)` | Train form: reuses the per-model 2-arg `preprocess` for `x` and lag-trims the target to `y = get_values(target)[:, nlags:end]`. |
| `postprocess!` | Writes the 2-D `y` into `output["surge"].values`. |

`forward(::AbstractFluxModel, x) = get_flux_model(m)(x)` and the single generic
`train_model!` (see the `AbstractFluxModel` section) serve every surge model —
single-input (`LinearSurgeModel`, `ConvSurgeModel`) and multi-input
(`AttentionSurgeModel`) alike.

### Input key handling

Data extraction accepts either `"stress_x"`/`"stress_y"` (used directly) or
`"wind_x"`/`"wind_y"` (converted via `uv_to_stress_xy`).  The helper
`_get_stress(input)` encapsulates this.

```
AbstractModel
    └── AbstractFluxModel   — predict, save_params, load_params!, forward, train_model!
            └── AbstractSurgeModel  — _surge_lag_windows, preprocess(…,target), postprocess!
                    ├── LinearSurgeModel     — preprocess
                    ├── ConvSurgeModel       — preprocess
                    └── AttentionSurgeModel  — preprocess
```

## LinearSurgeModel (`LinearSurgeModel.jl`)

`LinearSurgeModel <: AbstractSurgeModel` is the simplest concrete implementation,
using a single `Dense` layer (identity activation — linear regression) to
predict storm surge from wind-stress and pressure history.

### Settings

```julia
settings = Dict{String, Any}(
    "nlocations_output" => 5,   # number of output (surge) stations
    "nlocations_input"  => 9,   # number of input (forcing) locations
    "nlags"             => 16,  # number of previous time steps used as input
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
preprocess → x = (x_flat,)  with x_flat (3*nwind*nlags, ntimes_valid)
                + output Dict("surge" => zeros TimeSeries)
forward    → Chain(only, Dense(3*nwind*nlags => nstations))(x) → (nstations, ntimes_valid)   [generic, m(x)]
postprocess! → output["surge"].values .= y   [generic, 2-D]
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

## ConvSurgeModel (`ConvSurgeModel.jl`)

`ConvSurgeModel <: AbstractSurgeModel` applies 1-D convolutions over the lag
dimension of the wind-stress and pressure history.

### Constructor

```julia
model = ConvSurgeModel(settings::Dict{String, Any})
```

Required keys: `"nlocations_output"`, `"nlocations_input"`, `"nlags"`.  Optional `"model_pars"`:

| Key | Default | Description |
|---|---|---|
| `"channels"` | `[32, 16]` | Output channels per Conv1D layer |
| `"filtersize"` | `3` | Conv1D kernel width **and stride** (`stride = filtersize`): non-overlapping windows that shrink the lag length by `cld(·, filtersize)` per layer |
| `"activation"` | `"swish"` | Conv1D activation (`"swish"` or `"relu"`) |

### Data flow

`preprocess` builds the conv-ready `(lag, channel, batch-time)` tensor directly,
so `forward` (generic) runs the chain with no internal reshape:

```
preprocess → x = (xc,)  with xc (nlags, 3*nwind, ntimes_valid)   [lag, channel, batch]
forward    → Chain(only, Conv1D × N, flatten, Dense)(x)          [generic, m(x); only unwraps (xc,)]
             (Conv act, stride=filtersize, SamePad)
             → Dense(nlags_out*channels[end] → nstations)
             → (nstations, ntimes_valid)
postprocess! → output["surge"].values .= y   [generic, 2-D]
```

Overrides `preprocess` (conv-ready assembly); inherits `forward`,
`postprocess!`, `train_model!`, `save_params`, and `load_params!` from
`AbstractSurgeModel` / `AbstractFluxModel`.

```
AbstractModel
    └── AbstractFluxModel   — predict, save_params, load_params!
            └── AbstractSurgeModel  — _surge_lag_windows, forward, postprocess!, train_model!
                    ├── LinearSurgeModel     — preprocess
                    ├── ConvSurgeModel       — preprocess
                    └── AttentionSurgeModel  — preprocess
```

## AttentionSurgeModel (`AttentionSurgeModel.jl`)

`AttentionSurgeModel <: AbstractSurgeModel` uses a transformer branch network
for wind/pressure history and a dense trunk network for station metadata, merged
via graph-adjacency weights.

### Constructor

```julia
model = AttentionSurgeModel(settings::Dict{String, Any}, gn::GraphNetwork)
```

Required keys in `settings`: `"nlocations_output"`, `"nlocations_input"`, `"nlags"`, `"model_pars"`.

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

`AttentionSurgeModel` overrides only `preprocess` (it has two input tensors);
`forward` and `train_model!` come from `AbstractFluxModel`, `postprocess!` from
`AbstractSurgeModel` — the shared loop batches the input tuple element-wise:

| Method | Notes |
|---|---|
| `preprocess` | Returns `((x_station, x_wind), output)` — dual-input tuple |
| `forward` | Inherited generic `m(x)`. `AttentionSurgeFlux` carries a `(m)(x::Tuple)=m(x...)` method, then runs `AttentionSurgeFlux(x_station, x_wind)`, returning the 2-D last-lag slice directly |
| `train_model!` | Inherited generic loop |

### Data flow

The last-lag slice now lives inside `AttentionSurgeFlux`, so the flux model
returns the 2-D `(nstations, ntimes)` output directly:

```
preprocess → x = (x_station, x_wind)
             x_station (6, nstations, ntimes_valid)   [cos/sin lat, lon, day-of-year]
             x_wind    (3*nwind, nlags, ntimes_valid)
forward    → AttentionSurgeFlux(x_station, x_wind) → (nstations, nlags, ntimes)
             → last-lag slice (inside the flux model) → (nstations, ntimes_valid)   [generic]
postprocess! → output["surge"].values .= y   [generic, 2-D]
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
| `preprocess` (predict) | Builds `(4, nstations, ntimes)` station tensor and `(2*nfreqs, ntimes)` Doodson tensor; pre-allocates output |
| `preprocess` (train) | Reuses the predict form for `x`; `y = get_values(target["waterlevel"])` — the full series, no lag, no normalisation |
| `postprocess!` | Writes the 2-D `y` `(nstations, ntimes)` into `output["waterlevel"].values` |

`forward` and `train_model!` are inherited generically from `AbstractFluxModel`.
Both tide flux models (`TideModel`, `ProductTideFlux`) carry a
`(m)(x::Tuple)=m(x...)` method so the generic `m(x)` call reaches their 2-arg
implementation.

### Input convention

Both `input` and `target` carry a `"waterlevel"` key.  At training time they
point to the same `TimeSeries`.

```
AbstractModel
    └── AbstractFluxModel   — predict, save_params, load_params!, forward, train_model!
            └── AbstractTideModel  — preprocess, preprocess(…,target), postprocess!
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
preprocess → x = (x_station, x_doodson)
             x_station (4, nstations, ntimes)   [cos/sin lat, cos/sin lon]
             x_doodson (2*nfreqs, ntimes)        [cos/sin Doodson arguments]
forward    → TideModel(x_station, x_doodson) → (nstations, ntimes)   [generic m(x) via tuple method]
postprocess! → output["waterlevel"].values .= y   [generic, 2-D]
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

## write_outputs (`abstract_flux_model.jl` + `plot_utils.jl`)

Output generation is handled by `write_outputs`, a single implementation shared
by all `AbstractFluxModel` subtypes (it replaced the old per-model `plot_series`
when the pipeline became TOML-driven).  It is driven entirely by the
`[output_settings]` TOML section —
see `docs/output_settings.md` for the full schema.

```julia
write_outputs(model::AbstractFluxModel, data::Dict, all_settings::Dict)
```

For each configured output entry it calls `predict(model, …)` on the selected
split and dispatches to the relevant helper: `_plot_station_series`
(timeseries), `_plot_station_fft`, `_plot_station_scatter`,
`_write_station_stats`, `_write_station_series`, and — for tide models —
`_plot_station_tidal_analysis`.  A `summary.toml` with per-split RMSE is written
when `write_summary` is set.

The shared plotting skeleton `_plot_station_series` (in `src/plot_utils.jl`)
aligns target to prediction times (handles lag trimming via `select_timespan`),
computes per-station RMSE, and saves one PNG per station using
`Plots.plot(ts; location_index=i)` from `MultiTimeSeries.jl`.

## ProductTideModel (`ProductTideModel.jl`)

`ProductTideModel <: AbstractTideModel` uses a multiplicative product of learned
station and Doodson encodings, followed by residual gating layers.  Inspired by
the `TideInputLayer`/`TideLayer` architecture from the old `tides.jl`, adapted
to use the 4-feature cos/sin lat/lon station encoding instead of one-hot indices.

### Constructor

```julia
model = ProductTideModel(settings::Dict{String, Any})
```

Required key: `"freqs"`. Optional `"model_pars"`:

| Key | Default | Description |
|---|---|---|
| `"nfeats"` | `64` | Feature dimension throughout |
| `"nlayers"` | `3` | Number of `ProductGatingLayer`s |

### Data flow

```
preprocess → x_station (4, nstations, ntimes)   [cos/sin lat, cos/sin lon]
             x_doodson (2*nfreqs, ntimes)        [cos/sin Doodson arguments]

ProductInputLayer:
    Dense(4 → nfeats, identity; bias=false) applied to x_station
    Dense(2*nfreqs → nfeats, identity; bias=false) applied to x_doodson
    element-wise product → (nfeats, nstations, ntimes)

ProductGatingLayer × nlayers:
    x + Dense(nfeats → nfeats, relu)(x) * x

Dense(nfeats → 1) → (1, nstations, ntimes) → drop feature axis → (nstations, ntimes)
```

Inherits `preprocess`, `postprocess!`, `train_model!`, `save_params`, and
`load_params!` from `AbstractTideModel` / `AbstractFluxModel` without override.

```
AbstractModel
    └── AbstractFluxModel   — predict, save_params, load_params!, forward, train_model!
            └── AbstractTideModel  — preprocess, preprocess(…,target), postprocess!
                    ├── DeepONetTideModel
                    └── ProductTideModel
```

## Notes

- The new model hierarchy (`AbstractFluxModel` and subtypes) coexists with
  `TideSettings`/`WaveSettings`/`InteractionSettings` in `training.jl`.
  The old models will be migrated incrementally.
- `src/surge.jl` has been removed — all surge functionality is in the new
  `AbstractSurgeModel` hierarchy.
