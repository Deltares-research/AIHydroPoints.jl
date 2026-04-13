
# Software design

This file explores a future design for the model code, with the goal of creating a more modular and reusable codebase. The main ideas is to make models object-like, with a common interface for all models, and to separate the training settings from the model settings. This will make it easier to reuse the model code for different training settings, and it will make it easier to run inference with the trained model. We're mostly interested in ML models, but the design should be flexible enough to accommodate other types of models as well.

The model type here is quite basic and generic, and assumes that a model has input variables and output variables, where each variable is a collection of time series for different stations. These locations can but need not be the same for each variable. The model computations are causal and have a limited memory, so the output at a given time step only depends on the input at the current and previous time steps.

The main datatypes are:
- `FooModel<:AbstractModel`: holds the model settings and parameters for a specific model. Concrete subtypes are used to implement the different models, all using the same interface.
- `AbstractModelSettings::Dict{String,Any}`: holds the settings that are needed for training a model or for inference with a trained model. 
- `TrainingSettings::Dict{String,Any}`: holds the training settings that are needed for training.
- `DataSettings::Dict{String,Any}`: holds the settings for loading the training and test data.
- `InferenceSettings::Dict{String,Any}`: holds the settings for running inference with a trained model.

## Model design

A model is a data structure holding the settings and parameters of a specific model implementation. Its type is a subtype `ConcreteModel` of `AbstractModel`.
Concrete subtypes must implement the following methods:

- `predict(model::AbstractModel, input::Dict{String,TimeSeries}) -> Dict{String,TimeSeries}` Runs inference on the given input. Output variables may be defined on different locations than the input variables.
- `ConcreteModel(settings::ModelSettings) -> ConcreteModel` Constructs an uninitialised model (no parameters) from a settings dictionary.
- `get_settings(model::AbstractModel) -> ModelSettings` Returns the model settings required to reconstruct the model structure. Does not include trained parameters.
- `save_params(model::AbstractModel, file::String)` Saves the trained model parameters to a file.
- `load_params!(model::AbstractModel, file::String)` Loads trained parameters from a file into an existing model instance, mutating it in place.

### Train is kept separate from the model, but can be implemented as a method that dispatches on the model type. For example:
- `train_model!(model::AbstractModel, train_settings::TrainingSettings)` Trains the model on the data described in `train_settings`. 


**where and how**
Documentation fo the API can be done in src/model/abstract_model.jl. For example:
```julia
"""
    predict(model::AbstractModel, input::Dict{String,TimeSeries}) -> Dict{String,TimeSeries}

Run inference on `input` and return the model output. Both input and output are
dictionaries mapping variable names to time series, which may be defined on
different sets of locations.

# Arguments
- `model`: any concrete subtype of `AbstractModel`
- `input`: dictionary mapping variable names to input time series

# Notes
- Computations are causal: output at time `t` depends only on input at times `t` and earlier.
- Output locations need not match input locations.

Must be implemented by all concrete subtypes of `AbstractModel`.
"""
function predict(model::AbstractModel, input::Dict{String,TimeSeries})
    error("predict not implemented for $(typeof(model))")
end
```