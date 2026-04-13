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

## Required interface

Each concrete subtype `M <: AbstractModel` must pair with a concrete subtype
`S <: AbstractModelSettings` and implement:

| Function | Signature | Purpose |
|---|---|---|
| `prepare_train_data` | `(data::Dict, s::S) -> tuple` | Convert raw TimeSeries dicts into model-ready arrays |
| `train_epoch!` | `(m::M, s::S, loader, opt_state) -> loss` | One gradient-descent epoch |
| `compute_loss` | `(m::M, s::S, data) -> scalar` | Evaluation metric (used for val / early-stopping) |
| `predict` | `(m::M, s::S, data::Dict) -> TimeSeries` | Inference on new data |
| `plot_series` | `(m::M, s::S, data::Dict, name)` | Diagnostic plots |

The generic `train_model` loop in `training.jl` calls these through dispatch on
the settings type.  Once concrete models are made subtypes of `AbstractModel`,
dispatch can be moved to the model type instead.

## Optional interface

| Function | Default | Purpose |
|---|---|---|
| `save_model` | saves `Flux.state(model)` to JLD2 | Persist weights |
| `load_model` | loads state from JLD2 | Restore weights |
"""
abstract type AbstractModel end
