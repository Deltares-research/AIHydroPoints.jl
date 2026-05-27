using Test
using AIHydroPoints
using Dates

# ── Helpers ───────────────────────────────────────────────────────────────────

function mr_make_ts(nlocations, ntimes, quantity)
    values = randn(Float32, nlocations, ntimes)
    times  = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names  = ["loc$i" for i in 1:nlocations]
    lons   = Float64.(1:nlocations)
    lats   = Float64.(51 .+ (1:nlocations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

function mr_augmented_settings(model_name; nlags=4)
    ms = Dict{String,Any}(
        "model_name" => model_name,
        "model_dir"  => "test_dir",
        "nlags"      => nlags,
        "in_lats"    => [51.0, 52.0, 53.0],
        "in_lons"    => [1.0,  2.0,  3.0],
        "out_lats"   => [51.0, 52.0],
        "out_lons"   => [1.0,  2.0],
        "nlocations_input"  => 3,
        "nlocations_output" => 2,
    )
    return ms
end

# ── get_model_type ────────────────────────────────────────────────────────────

@testset "get_model_type happy path" begin
    @test get_model_type(Dict("model_name" => "LinearSurgeModel"))    === LinearSurgeModel
    @test get_model_type(Dict("model_name" => "ConvSurgeModel"))      === ConvSurgeModel
    @test get_model_type(Dict("model_name" => "AttentionSurgeModel")) === AttentionSurgeModel
    @test get_model_type(Dict("model_name" => "DeepONetTideModel"))   === DeepONetTideModel
    @test get_model_type(Dict("model_name" => "ProductTideModel"))    === ProductTideModel
    @test get_model_type(Dict("model_name" => "ConvWaveModel"))       === ConvWaveModel
    @test get_model_type(Dict("model_name" => "DeepONetWaveModel"))   === DeepONetWaveModel
    @test get_model_type(Dict("model_name" => "ConvInteractionModel"))    === ConvInteractionModel
    @test get_model_type(Dict("model_name" => "ProductInteractionModel")) === ProductInteractionModel
end

@testset "get_model_type errors" begin
    # missing key
    @test_throws ErrorException get_model_type(Dict{String,Any}())
    # unknown name — error message should mention the name
    err = try get_model_type(Dict("model_name" => "BogusModel")); nothing
          catch e; e; end
    @test err isa ErrorException
    @test occursin("BogusModel", err.msg)
    @test occursin("Known models", err.msg)
end

# ── validate_model_settings! default no-op ───────────────────────────────────

@testset "validate_model_settings! default no-op" begin
    ms = Dict{String,Any}("model_name" => "LinearSurgeModel")
    @test isnothing(validate_model_settings!(LinearSurgeModel, ms))
end

# ── create_model ─────────────────────────────────────────────────────────────

@testset "create_model LinearSurgeModel" begin
    ms = mr_augmented_settings("LinearSurgeModel")
    model = create_model(ms, Dict{String,TimeSeries}())
    @test model isa LinearSurgeModel
end

@testset "create_model ConvSurgeModel" begin
    ms = mr_augmented_settings("ConvSurgeModel")
    ms["model_pars"] = Dict{String,Any}("channels" => [16, 1], "filtersize" => 3)
    model = create_model(ms, Dict{String,TimeSeries}())
    @test model isa ConvSurgeModel
end

@testset "create_model AttentionSurgeModel builds GraphNetwork from settings" begin
    ms = mr_augmented_settings("AttentionSurgeModel")
    ms["model_pars"] = Dict{String,Any}(
        "nembed" => 8, "theta" => 1000.0, "nheads" => 2,
        "nlayers_branch" => 1, "nlayers_trunk" => 1, "nhidden_trunk" => 8,
    )
    model = create_model(ms, Dict{String,TimeSeries}())
    @test model isa AttentionSurgeModel
end

@testset "create_model unknown name errors" begin
    ms = Dict{String,Any}("model_name" => "NoSuchModel")
    @test_throws ErrorException create_model(ms, Dict{String,TimeSeries}())
end

# ── validate_and_augment_settings! calls the hook ────────────────────────────

# Dummy model type used only to test the validate_model_settings! hook,
# so we never overwrite methods on real model types.
struct _TestHookModel <: AIHydroPoints.AbstractSurgeModel end
AIHydroPoints.MODEL_REGISTRY["_TestHookModel"] = _TestHookModel

hook_called = Ref(false)
AIHydroPoints.validate_model_settings!(::Type{_TestHookModel}, ::Dict) =
    (hook_called[] = true; nothing)

@testset "validate_and_augment_settings! calls validate_model_settings!" begin
    hook_called[] = false

    all_settings = Dict{String,Any}(
        "run_info"       => Dict{String,Any}("runid" => "test"),
        "model_settings" => Dict{String,Any}("model_name" => "_TestHookModel", "nlags" => 4),
        "train_settings" => Dict{String,Any}(),
        "data_settings"  => Dict{String,Any}("model_io" => Dict("input" => ["stress_x"], "target" => ["surge"])),
    )
    train_input  = Dict{String,TimeSeries}("stress_x" => mr_make_ts(2, 20, "stress_x"))
    train_target = Dict{String,TimeSeries}("surge"    => mr_make_ts(2, 20, "surge"))

    validate_and_augment_settings!(all_settings, train_input, train_target)
    @test hook_called[]
end
