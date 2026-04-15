
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

## Model design

A model is a data structure holding the settings and parameters of a specific
model implementation. Its type is a concrete subtype `ConcreteModel` of
`AbstractModel`.

Concrete subtypes must implement the following methods:

- `ConcreteModel(settings::Dict{String, Any}) -> ConcreteModel` — Constructs an uninitialised model (random or zero weights) from a settings dictionary. Settings are stored inside the model.
- `predict(model::AbstractModel, input::Dict{String, TimeSeries}) -> Dict{String, TimeSeries}` — Runs inference on the given input. `TimeSeries` is the concrete type from `MultiTimeSeries.jl`. Output variables may be at different locations than input variables.
- `get_settings(model::AbstractModel) -> Dict{String, Any}` — Returns the settings required to reconstruct the model structure. Does not include trained parameters.
- `save_params(model::AbstractModel, file::String)` — Serialises trained weights to a file (not settings).
- `load_params!(model::AbstractModel, file::String)` — Loads trained weights from a file into an existing model instance in-place.

Training is kept separate from the model, dispatching on the model type:
- `train_model!(model::AbstractModel, train_settings::TrainingSettings)` — Trains the model in-place. Returns training diagnostics (losses, etc.) in a form decided by the concrete implementation.

### AbstractFluxModel

AbstractFluxModel is an abstract subtype of `AbstractModel` that wraps a Flux.jl model. It provides a common interface for training and inference with Flux models, and can be used as a base type for all ML models in the codebase. It is not intended to be used directly, but rather to be subclassed by specific model implementations (e.g. `TideModel`, `SurgeModel`, `WaveModel`), which will implement the required methods and add any additional functionality needed for their specific use case.

`AbstractFluxModel <: AbstractModel`

A subtype of `AbstractModel` that implements models by wrapping a Flux model in a time-series context. It sits between `AbstractModel` and concrete model types, implementing the generic Flux machinery once while leaving the data mapping and model architecture as customisation points.

```
AbstractModel
    └── AbstractFluxModel   — implements predict, handles pre/postprocessing and Flux machinery
            └── FooModel    — implements preprocess, postprocess, forward and model-specific settings
```

### What `AbstractFluxModel` implements

`predict` is implemented once at this level:

```julia
function predict(model::AbstractFluxModel, input::Dict{String,TimeSeries})
    tensor = preprocess(model, input)    # Dict -> (locations, features, time, batch)
    output = forward(model, tensor)      # model-specific reshape + Flux forward pass
    return postprocess(model, output)    # tensor -> Dict
end
```

### Interface — concrete subtypes must implement

- `preprocess(model::AbstractFluxModel, input::Dict{String,TimeSeries}) -> Array{Float32,4}`  
  Maps input time series to a tensor of shape `(locations, features, time, batch)`. Responsible for variable/location ordering, scaling, and any other input transformations.

- `postprocess(model::AbstractFluxModel, output::Array) -> Dict{String,TimeSeries}`  
  Maps the Flux output tensor back to named time series. Applies inverse scaling and any output transformations. Note that `postprocess(preprocess(x))` is not expected to recover `x` in general.

- `forward(model::AbstractFluxModel, tensor::Array{Float32,4}) -> Array{Float32,3}`  
  Reshapes the tensor as needed and runs the Flux forward pass. The reshape depends on the model architecture:

```julia
# For a dense model
x_flat = reshape(x, locations*features*time, batch)

# For a 1D time convolution
x_flat = reshape(x, locations*features, time, batch)
```

### Tensor layout

The canonical tensor format for inputs at this level is `(locations, features, time_lag, time)`. This layout is chosen so that the first dimensions are contiguous in memory under Julia's column-major convention, making reshapes for both dense and convolutional models allocation-free.
For the output of `forward`, the canonical format is `(locations, features, time)`. We're assuming that the output for a single time step depends on a fixed number of previous time steps (the time lag). Therefore, we can use the output time as the batch dimension for training and inference.


## Notes

- The work in `src/models/` is a prototype for the future model interface and not necessarily fully consistent with the existing concrete models (`TideModel`, `SurgeModel`, `WaveSettings`). Integration is planned in steps 5e–5f of `plan.md`.
- The existing `train_model` (no `!`) in `training.jl` and the existing settings structs (`TideSettings`, etc.) are not yet subtypes of `AbstractModel`; they will be migrated incrementally.
- Where `train_model!` receives training data is left to the concrete implementation — the top-level interface only specifies the model and training settings.
