# model_registry.jl
#
# Central registry that maps model_name strings to Julia types.
# Provides get_model_type, validate_model_settings!, and create_model.

const MODEL_REGISTRY = Dict{String,Type}(
    "LinearSurgeModel"     => LinearSurgeModel,
    "ConvSurgeModel"       => ConvSurgeModel,
    "AttentionSurgeModel"  => AttentionSurgeModel,
    "DeepONetTideModel"    => DeepONetTideModel,
    "ProductTideModel"     => ProductTideModel,
    "ConvWaveModel"        => ConvWaveModel,
    "DeepONetWaveModel"    => DeepONetWaveModel,
    "ConvInteractionModel"    => ConvInteractionModel,
    "ProductInteractionModel" => ProductInteractionModel,
    "BiLinearSurgeInteractionModel" => BiLinearSurgeInteractionModel,
)

"""
    get_model_type(model_settings::Dict) -> Type

Look up the Julia type for `model_settings["model_name"]` in `MODEL_REGISTRY`.
Errors with a descriptive message when the key is missing or the name is unknown.
"""
function get_model_type(model_settings::Dict)
    haskey(model_settings, "model_name") ||
        error("get_model_type: model_settings missing \"model_name\" key")
    name = model_settings["model_name"]
    return get(MODEL_REGISTRY, name) do
        known = join(sort(collect(keys(MODEL_REGISTRY))), ", ")
        error("get_model_type: unknown model_name \"$name\". Known models: $known")
    end
end

"""
    validate_model_settings!(::Type{T}, model_settings::Dict) where T <: AbstractModel

Model-specific settings validation hook, called from `validate_and_augment_settings!`
after generic augmentation is complete.  The default method is a no-op; concrete model
types may add methods to enforce model-specific required keys.
"""
validate_model_settings!(::Type{T}, ::Dict) where {T<:AbstractModel} = nothing

"""
    create_model(model_settings::Dict, train_input::Dict{String,TimeSeries}) -> AbstractModel

Factory: look up the model type from `model_settings["model_name"]` and construct it.
Must be called after `validate_and_augment_settings!` so that location metadata
(`in_lats`, `in_lons`, `out_lats`, `out_lons`) is available in `model_settings`.
"""
create_model(ms::Dict, train_input::Dict{String,TimeSeries}) =
    create_model(get_model_type(ms), ms, train_input)

create_model(::Type{LinearSurgeModel},    ms, _) = LinearSurgeModel(ms)
create_model(::Type{ConvSurgeModel},      ms, _) = ConvSurgeModel(ms)
create_model(::Type{DeepONetTideModel},   ms, _) = DeepONetTideModel(ms)
create_model(::Type{ProductTideModel},    ms, _) = ProductTideModel(ms)
create_model(::Type{ConvWaveModel},       ms, _) = ConvWaveModel(ms)
create_model(::Type{DeepONetWaveModel},   ms, _) = DeepONetWaveModel(ms)
create_model(::Type{ConvInteractionModel},   ms, _) = ConvInteractionModel(ms)
create_model(::Type{ProductInteractionModel},ms, _) = ProductInteractionModel(ms)
create_model(::Type{BiLinearSurgeInteractionModel}, ms, _) = BiLinearSurgeInteractionModel(ms)

function create_model(::Type{AttentionSurgeModel}, ms, _)
    in_points  = collect(zip(ms["in_lats"],  ms["in_lons"]))
    out_points = collect(zip(ms["out_lats"], ms["out_lons"]))
    gn = GraphNetwork(in_points, out_points)
    return AttentionSurgeModel(ms, gn)
end
