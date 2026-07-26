using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (uniquely named to avoid clashing with other model test files)
# ──────────────────────────────────────────────────────────────────────────────

function _si_ts(values::Matrix{Float32}, quantity::String)
    nstations, ntimes = size(values)
    times = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names = ["s$i" for i in 1:nstations]
    lons  = Float64.(1:nstations)
    lats  = Float64.(51 .+ (1:nstations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

"""Forcing (nwind stations) + tide (nstations output stations) input dict."""
function _si_input(; nwind=3, nstations=2, ntimes=30)
    sx   = _si_ts(randn(Float32, nwind, ntimes), "stress_x")
    sy   = _si_ts(randn(Float32, nwind, ntimes), "stress_y")
    pr   = _si_ts(ones(Float32, nwind, ntimes) .* 1.0f5, "pressure")
    tide = _si_ts(randn(Float32, nstations, ntimes), "tide")
    return Dict{String, TimeSeries}(
        "stress_x" => sx, "stress_y" => sy, "pressure" => pr, "tide" => tide)
end

_si_target(; nstations=2, ntimes=30) =
    Dict{String, TimeSeries}("surge" => _si_ts(randn(Float32, nstations, ntimes), "surge"))

function _si_settings(; nstations=2, nwind=3, nlags=4, model_pars=nothing)
    s = Dict{String, Any}(
        "nlocations_output" => nstations,
        "nlocations_input"  => nwind,
        "nlags"             => nlags,
        "out_names"    => ["s$i" for i in 1:nstations],
        "out_lons"     => Float64.(1:nstations),
        "out_lats"     => Float64.(51 .+ (1:nstations)),
        "out_quantity" => "surge",
    )
    model_pars !== nothing && (s["model_pars"] = model_pars)
    return s
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────

@testset "BiLinearSurgeInteractionModel construction" begin
    m = BiLinearSurgeInteractionModel(
        Dict{String, Any}("nlocations_output" => 2, "nlocations_input" => 3, "nlags" => 4))

    @test m isa AbstractSurgeInteractionModel
    @test m isa AbstractSurgeModel
    @test m isa AbstractFluxModel

    fx = get_flux_model(m)
    @test fx isa BiLinearSurgeInteractionFlux
    @test fx.surge isa Dense
    @test size(fx.surge.weight) == (2, 3*3*4)     # nstations × 3·nwind·nlags
    @test size(fx.mod.V) == (2, 4)                # (nstations, nlags)
    @test all(fx.mod.V .== 0f0)                   # zero-init modulation
    @test fx.mod.a == 0.1f0
    @test fx.mod.σ === identity
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — two-tensor tuple (f_flat, t_lags)
# ──────────────────────────────────────────────────────────────────────────────

@testset "BiLinearSurgeInteractionModel preprocess" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m = BiLinearSurgeInteractionModel(_si_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    nvalid = ntimes - nlags + 1

    (f_flat, t_lags), output = preprocess(m, _si_input(nwind=nwind, nstations=nstations, ntimes=ntimes))

    @test size(f_flat) == (3*nwind*nlags, nvalid)
    @test size(t_lags) == (nlags, nstations, nvalid)
    @test eltype(f_flat) == Float32
    @test eltype(t_lags) == Float32
    @test haskey(output, "surge")
    @test size(output["surge"].values) == (nstations, nvalid)
    @test all(output["surge"].values .== 0f0)
end

@testset "BiLinearSurgeInteractionModel preprocess errors on missing tide" begin
    m = BiLinearSurgeInteractionModel(_si_settings())
    input = _si_input()
    delete!(input, "tide")
    @test_throws ErrorException preprocess(m, input)
end

# ──────────────────────────────────────────────────────────────────────────────
# forward — zero-init modulation ≡ 1  (starts as LinearSurgeModel)
# ──────────────────────────────────────────────────────────────────────────────

@testset "BiLinearSurgeInteractionModel forward; zero-init modulation ≡ 1" begin
    nwind = 3; nstations = 2; nlags = 4; ntimes = 20
    m = BiLinearSurgeInteractionModel(_si_settings(nstations=nstations, nwind=nwind, nlags=nlags))

    f_flat = randn(Float32, 3*nwind*nlags, ntimes)
    t_lags = randn(Float32, nlags, nstations, ntimes)

    y = forward(m, (f_flat, t_lags))
    @test size(y) == (nstations, ntimes)
    @test eltype(y) == Float32

    # zero V ⇒ modulation exactly 1 ⇒ output equals the surge branch alone
    modulation = get_flux_model(m).mod(t_lags)
    @test all(modulation .== 1f0)
    @test y ≈ get_flux_model(m).surge(f_flat)
end

# ──────────────────────────────────────────────────────────────────────────────
# model_pars: a and mod_activation
# ──────────────────────────────────────────────────────────────────────────────

@testset "BiLinearSurgeInteractionModel mod_activation / a" begin
    m = BiLinearSurgeInteractionModel(
        _si_settings(model_pars=Dict{String,Any}("mod_activation" => "tanh", "a" => 0.2)))
    @test get_flux_model(m).mod.σ === tanh
    @test get_flux_model(m).mod.a == 0.2f0
    # tanh(0) = 0 ⇒ modulation still 1 at zero-init
    t_lags = randn(Float32, 4, 2, 5)
    @test all(get_flux_model(m).mod(t_lags) .== 1f0)

    @test_throws ErrorException BiLinearSurgeInteractionModel(
        _si_settings(model_pars=Dict{String,Any}("mod_activation" => "nope")))
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — metadata population + training loop
# ──────────────────────────────────────────────────────────────────────────────

@testset "BiLinearSurgeInteractionModel train_model!" begin
    nwind = 3; nstations = 2; ntimes = 50; nlags = 4
    settings = Dict{String, Any}(
        "nlocations_output" => nstations, "nlocations_input" => nwind, "nlags" => nlags)
    m      = BiLinearSurgeInteractionModel(settings)
    input  = _si_input(nwind=nwind, nstations=nstations, ntimes=ntimes)
    target = _si_target(nstations=nstations, ntimes=ntimes)

    ts = TrainingSettings(nepochs=3, batch_size=16, learning_rate=1e-3)
    train_losses, val_losses = train_model!(m, ts, input, target)

    @test haskey(m.settings, "out_names")
    @test m.settings["out_names"] == get_names(target["surge"])
    @test length(train_losses) == ts.nepochs
    @test eltype(train_losses) == Float32
    @test all(train_losses .>= 0f0)
    @test isempty(val_losses)
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

@testset "BiLinearSurgeInteractionModel predict" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m     = BiLinearSurgeInteractionModel(_si_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    input = _si_input(nwind=nwind, nstations=nstations, ntimes=ntimes)

    output = predict(m, input)
    nvalid = ntimes - nlags + 1

    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "surge")
    @test size(output["surge"].values) == (nstations, nvalid)
    @test !all(output["surge"].values .== 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip (incl. the custom modulation weights)
# ──────────────────────────────────────────────────────────────────────────────

@testset "BiLinearSurgeInteractionModel save/load params" begin
    m  = BiLinearSurgeInteractionModel(_si_settings())
    fx = get_flux_model(m)

    fx.mod.V .= randn(Float32, size(fx.mod.V))   # perturb so there's something to round-trip
    V_orig = copy(fx.mod.V)
    W_orig = copy(fx.surge.weight)

    fn = joinpath(temp_dir, "bilinear_surge_interaction_params.jld2")
    save_params(m, fn)
    @test isfile(fn)
    @test_throws ErrorException save_params(m, fn)          # exists, overwrite=false
    save_params(m, fn; overwrite=true)

    fx.mod.V       .= 0f0
    fx.surge.weight .= 0f0
    load_params!(m, fn)

    @test get_flux_model(m).mod.V       ≈ V_orig
    @test get_flux_model(m).surge.weight ≈ W_orig
end
