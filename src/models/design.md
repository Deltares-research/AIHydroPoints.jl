
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
| `save_params`  | `(m::M, file::String)` | Serialise trained weights to file |
| `load_params!` | `(m::M, file::String)` | Load weights from file in-place |
| `train_model!` | `(m::M, train_settings::TrainingSettings)` | Train in-place |

`TimeSeries` is the concrete type from `MultiTimeSeries.jl`.
All methods have fallback implementations that throw a descriptive error when a
subtype forgets to implement them.

## AbstractFluxModel (`abstract_flux_model.jl`)

`AbstractFluxModel <: AbstractModel` sits between `AbstractModel` and concrete
Flux-based model types.  It implements `predict`, `save_params`, and
`load_params!` once, delegating to three customisation points and `get_flux_model`.

```
AbstractModel
    └── AbstractFluxModel   — implements predict, save_params, load_params!
            ├── MyFluxModel — convenience concrete struct (chain + settings dict)
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

### MyFluxModel

`MyFluxModel <: AbstractFluxModel` is a convenience concrete struct for cases
where domain models do not need their own struct.  It holds a Flux chain and a
settings dict, and provides `get_flux_model` and `get_settings` automatically.

```julia
model = MyFluxModel(my_chain, settings)

# Add domain behaviour via methods that dispatch on MyFluxModel:
preprocess(m::MyFluxModel, input) = ...
forward(m::MyFluxModel, x)        = ...
postprocess!(output, m::MyFluxModel, y) = ...
```

When a domain model needs additional fields (e.g. cached station metadata),
define it as its own `<: AbstractFluxModel` struct instead.

## LinearSurgeModel (`LinearSurgeModel.jl`)

`LinearSurgeModel <: AbstractFluxModel` is the first concrete implementation,
used to validate the interface.  It predicts storm surge from wind-stress and
pressure history using a single `Dense` layer (identity activation — a linear
regression).

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
| `"wind_x"`     | `(nwind, T)` | East wind-stress component |
| `"wind_y"`     | `(nwind, T)` | North wind-stress component |
| `"pressure"`   | `(nwind, T)` | Sea-level pressure (scaled: `2e-4*(p - 1e5)`) |
| `"waterlevel"` | `(nstations, T)` | Used only for output station metadata |

### Data flow

```
preprocess → tensor (1, 3*nwind, nlags, ntimes_valid)
                + output Dict("surge" => zeros TimeSeries)
forward    → flatten → Dense(3*nwind*nlags => nstations) → (nstations, 1, ntimes_valid)
postprocess! → output["surge"].values .= y[:, 1, :]
```

## Notes

- The work in `src/models/` is a prototype for the future model interface and
  not yet consistent with the existing concrete models (`TideModel`, `SurgeModel`,
  `WaveSettings`). Integration is planned in steps 5e–5f of `plan.md`.
- The existing `train_model` (no `!`) in `training.jl` and the existing settings
  structs (`TideSettings`, etc.) are not yet subtypes of `AbstractModel`; they
  will be migrated incrementally.
- `train_model!` signature leaves training data handling to the concrete
  implementation — the top-level interface only specifies the model and training
  settings.
