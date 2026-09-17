# serve.jl
#
# KServe V2 REST predictor for an AbstractFluxModel. Loads a trained model from
# MODEL_DIR (model_settings.toml + weights file) once at startup and exposes it
# over the Open Inference Protocol (KServe V2) via HTTP.jl.
#
# Wire contract (see AbstractSurgeModel.jl for what predict() itself requires):
#   - Input tensors: one per forcing variable predict() expects (e.g.
#     "stress_x"/"stress_y" or "wind_x"/"wind_y", and "pressure" for surge
#     models; interaction models additionally require "tide"), each shaped
#     [nlocations, ntimes] (column-major, Julia's native TimeSeries layout —
#     this is a bespoke server, not a Triton-wire-compatible one, so no
#     row-major flip is needed). Plus a shared "times" tensor of ISO8601
#     timestamp strings, one per timestep, common to every input tensor.
#   - Output tensors: one per key in predict()'s output Dict (e.g. "surge"),
#     each [nlocations, ntimes_valid], plus a "times" tensor holding the
#     valid output time axis — models like AbstractSurgeModel drop the first
#     nlags-1 input timesteps, so output times are NOT simply the input times.
#   - Station metadata (names/lons/lats) is not sent per-request: it's fixed
#     per deployed model version and read from model_settings.toml at load time.

using HTTP
using JSON3
using Dates: DateTime

"""
    load_served_model(model_dir::String) -> (model, settings)

Reconstruct a trained model from `model_dir` for inference — the same
`create_model` + `load_params!` steps `predict(input_toml::String)` uses, run
once at server startup. `model_dir` must directly contain `model_settings.toml`
and the weights file it names (as written by `train()`, or as staged by
KServe's `storageUri` download into e.g. `/mnt/models`).
"""
function load_served_model(model_dir::String)
    isdir(model_dir) || error("MODEL_DIR not found: $model_dir")
    settings_path = joinpath(model_dir, "model_settings.toml")
    isfile(settings_path) || error("model_settings.toml not found in $model_dir")
    settings = toml_read(settings_path)
    model = create_model(settings, Dict{String, TimeSeries}())
    weights_file = joinpath(model_dir, get(settings, "model_weights", "params.jld2"))
    isfile(weights_file) || error("Weights file not found: $weights_file")
    load_params!(model, weights_file)
    return model, settings
end

# ──────────────────────────────────────────────────────────────────────────────
# V2 tensor <-> TimeSeries wire adapter
# ──────────────────────────────────────────────────────────────────────────────

function _tensor_by_name(tensors, name::AbstractString)
    idx = findfirst(t -> String(t.name) == name, tensors)
    idx === nothing && error("Missing required input tensor \"$name\"")
    return tensors[idx]
end

"""
    _parse_times(tensors) -> Vector{DateTime}

Parse the shared "times" input tensor (ISO8601 strings) into `DateTime`s.
"""
_parse_times(tensors) = DateTime.(String.(_tensor_by_name(tensors, "times").data))

"""
    _tensor_to_timeseries(t, times, meta) -> TimeSeries

Build a `TimeSeries` from one request tensor. `meta` supplies the station
metadata (`names`, `lons`, `lats`) that is not part of the wire contract —
it comes from the loaded model's settings instead.
"""
function _tensor_to_timeseries(t, times::Vector{DateTime}, meta)
    nloc, ntime = t.shape[1], t.shape[2]
    length(times) == ntime ||
        error("Tensor \"$(t.name)\" has $ntime timesteps, \"times\" has $(length(times))")
    values = reshape(Float32.(t.data), nloc, ntime)
    return TimeSeries(values, times, meta.names, meta.lons, meta.lats, String(t.name), "kserve-request")
end

"""
    _timeseries_to_tensor(name, ts) -> Dict

Serialize a `TimeSeries`'s values into a V2 response tensor.
"""
function _timeseries_to_tensor(name::AbstractString, ts::TimeSeries)
    v = get_values(ts)
    return Dict(
        "name" => name,
        "shape" => collect(Int, size(v)),
        "datatype" => "FP32",
        "data" => vec(Float32.(v)),
    )
end

"""
    _times_tensor(name, times) -> Dict

Serialize a time axis (`Vector{DateTime}`) into a V2 response tensor of
ISO8601 strings.
"""
function _times_tensor(name::AbstractString, times::Vector{DateTime})
    return Dict(
        "name" => name,
        "shape" => [length(times)],
        "datatype" => "BYTES",
        "data" => string.(times),
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Station metadata (from model_settings.toml, not the wire contract)
# ──────────────────────────────────────────────────────────────────────────────

"""
    input_location_meta(settings) -> NamedTuple{(:names, :lons, :lats)}

Metadata for the model's *input* (forcing) locations. Falls back to the
output metadata for model families that don't persist a separate input grid.
"""
function input_location_meta(settings)
    names = get(settings, "in_names", settings["out_names"])
    lons  = get(settings, "in_lons",  settings["out_lons"])
    lats  = get(settings, "in_lats",  settings["out_lats"])
    return (; names, lons, lats)
end

# ──────────────────────────────────────────────────────────────────────────────
# HTTP handlers
# ──────────────────────────────────────────────────────────────────────────────

struct ServedModel
    model::Any
    settings::Dict{String, Any}
    name::String
end

function _error_response(status::Int, message::AbstractString)
    return HTTP.Response(
        status, ["Content-Type" => "application/json"];
        body = JSON3.write(Dict("error" => message)),
    )
end

function _handle_infer(req::HTTP.Request, served::ServedModel)
    local tensors
    try
        body = JSON3.read(req.body)
        tensors = body.inputs
    catch exc
        return _error_response(400, "Malformed request body: $exc")
    end

    local output
    try
        times = _parse_times(tensors)
        meta = input_location_meta(served.settings)
        input = Dict{String, TimeSeries}()
        for t in tensors
            String(t.name) == "times" && continue
            input[String(t.name)] = _tensor_to_timeseries(t, times, meta)
        end
        output = predict(served.model, input)
    catch exc
        return _error_response(400, "Inference failed: $exc")
    end

    out_times = get_times(first(values(output)))
    resp = Dict(
        "model_name" => served.name,
        "outputs" => vcat(
            [_timeseries_to_tensor(k, v) for (k, v) in output],
            [_times_tensor("times", out_times)],
        ),
    )
    return HTTP.Response(200, ["Content-Type" => "application/json"]; body = JSON3.write(resp))
end

function _build_router(served::ServedModel)
    router = HTTP.Router()
    HTTP.register!(router, "GET", "/v2/health/live", _ -> HTTP.Response(200))
    HTTP.register!(router, "GET", "/v2/health/ready", _ -> HTTP.Response(200))
    HTTP.register!(router, "GET", "/v2/models/$(served.name)/ready", _ -> HTTP.Response(200))
    HTTP.register!(router, "POST", "/v2/models/$(served.name)/infer", req -> _handle_infer(req, served))
    return router
end

"""
    serve(; model_dir=ENV["MODEL_DIR"], model_name=ENV["MODEL_NAME"], port=8080)

Load a trained model from `model_dir` and serve it over KServe's V2 REST
protocol on `port`, blocking forever. `model_name` defaults to the value in
`model_settings.toml` if set on the environment.
"""
function serve(;
        model_dir::String = ENV["MODEL_DIR"],
        model_name::Union{String, Nothing} = get(ENV, "MODEL_NAME", nothing),
        port::Int = parse(Int, get(ENV, "SERVER_PORT", "8080")),
    )
    model, settings = load_served_model(model_dir)
    name = something(model_name, get(settings, "model_name", "model"))
    served = ServedModel(model, settings, name)
    router = _build_router(served)
    @info "Serving \"$name\" on :$port (MODEL_DIR=$model_dir)"
    HTTP.serve(router, "0.0.0.0", port)
end
