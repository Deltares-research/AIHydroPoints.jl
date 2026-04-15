# abstract_model.jl
#
# Defines the AbstractModel type and the interface that every concrete model
# is expected to implement.
#
# Current concrete models (TideModel, SurgeModel, wave Chain, InteractionModel)
# are not yet subtypes of AbstractModel — this file is a staging area to agree
# on the interface before integrating it.

"""
    AbstractModel

Supertype for all AI-Hydro forecast models (tides, surge, interaction, waves).

A concrete subtype `M <: AbstractModel` is a self-contained object: it stores
its own settings and trained parameters.  The interface below is designed so
that inference scripts only need the model object — no separate settings struct
has to be passed around.

## Constructor convention

```julia
ConcreteModel(settings::Dict{String, Any}) -> ConcreteModel
```

Construct an uninitialised model (random or zero weights) from a settings
dictionary.  The settings are stored inside the model so they can be retrieved
later via `get_settings`.

## Required interface

| Function | Signature | Purpose |
|---|---|---|
| `predict` | `(m::M, input::Dict{String,TimeSeries}) -> Dict{String,TimeSeries}` | Unified inference entry point |
| `get_settings` | `(m::M) -> Dict{String, Any}` | Return the settings stored in the model |
| `save_params` | `(m::M, file::String)` | Serialise trained weights to file (not settings) |
| `load_params!` | `(m::M, file::String)` | Load weights from file into model in-place |

## Training interface

| Function | Signature | Purpose |
|---|---|---|
| `train_model!` | `(m::M, train_settings::TrainingSettings)` | Train the model in-place |

## Notes

- Input and output of `predict` are dicts mapping variable names to time series.
  Output locations need not match input locations.
- Computations are causal: output at time `t` depends only on input at times ≤ `t`.
- `save_params` / `load_params!` persist **only** trained weights; settings are handled
  separately by `save_settings` / `load_settings`.
- `train_model!` mutates the model in-place (updates weights) and returns training
  diagnostics (losses, etc.) decided by the concrete implementation.
"""
abstract type AbstractModel end

# ──────────────────────────────────────────────────────────────────────────────
# Interface fallbacks — throw a clear error when a subtype forgets to implement
# a required method.
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict(model::AbstractModel, input::Dict{String, TimeSeries})
        -> Dict{String, TimeSeries}

Run inference on `input` and return the model output as a dictionary of time
series.  Output locations need not match input locations.  Computations are
causal: output at time `t` depends only on input at times `t` and earlier.

Must be implemented by every concrete subtype of `AbstractModel`.
"""
function predict(model::AbstractModel, input::Dict{String, TimeSeries})
    error("predict not implemented for $(typeof(model))")
end

"""
    get_settings(model::AbstractModel) -> Dict{String, Any}

Return the inference-time settings stored inside `model` as a plain dictionary.
These are the same settings that were passed to the constructor and are
sufficient to reconstruct the model architecture (but not the trained weights).

Must be implemented by every concrete subtype of `AbstractModel`.
"""
function get_settings(model::AbstractModel)
    error("get_settings not implemented for $(typeof(model))")
end

"""
    save_params(model::AbstractModel, file::String)

Serialise the trained model weights to `file` (JLD2 format).
Settings are **not** included; use `save_settings` for those.

Must be implemented by every concrete subtype of `AbstractModel`.
"""
function save_params(model::AbstractModel, file::String)
    error("save_params not implemented for $(typeof(model))")
end

"""
    load_params!(model::AbstractModel, file::String)

Load trained weights from `file` into `model` in-place.
The model structure must already match the saved weights (i.e. build the model
from its settings first, then call `load_params!`).

Must be implemented by every concrete subtype of `AbstractModel`.
"""
function load_params!(model::AbstractModel, file::String)
    error("load_params! not implemented for $(typeof(model))")
end

"""
    train_model!(model::AbstractModel, train_settings::TrainingSettings)

Train `model` in-place using `train_data`, evaluating on `test_data`.
All training hyperparameters (epochs, learning rate, etc.) are taken from
`train_settings`.  Returns training diagnostics (e.g. loss history) in a
form decided by the concrete implementation.

Must be implemented by every concrete subtype of `AbstractModel`.
"""
function train_model!(model::AbstractModel, train_settings::TrainingSettings)
    error("train_model! not implemented for $(typeof(model))")
end
