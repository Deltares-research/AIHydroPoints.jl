using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (uniquely named to avoid clashing with other model test files)
# ──────────────────────────────────────────────────────────────────────────────

function _dsi_ts(values::Matrix{Float32}, quantity::String)
    nstations, ntimes = size(values)
    times = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names = ["s$i" for i in 1:nstations]
    lons  = Float64.(1:nstations)
    lats  = Float64.(51 .+ (1:nstations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

"""Forcing (nwind stations) + tide (nstations output stations) input dict."""
function _dsi_input(; nwind=3, nstations=2, ntimes=30)
    sx   = _dsi_ts(randn(Float32, nwind, ntimes), "stress_x")
    sy   = _dsi_ts(randn(Float32, nwind, ntimes), "stress_y")
    pr   = _dsi_ts(ones(Float32, nwind, ntimes) .* 1.0f5, "pressure")
    tide = _dsi_ts(randn(Float32, nstations, ntimes), "tide")
    return Dict{String, TimeSeries}(
        "stress_x" => sx, "stress_y" => sy, "pressure" => pr, "tide" => tide)
end

_dsi_target(; nstations=2, ntimes=30) =
    Dict{String, TimeSeries}("surge" => _dsi_ts(randn(Float32, nstations, ntimes), "surge"))

function _dsi_settings(; nstations=2, nwind=3, nlags=4, model_pars=nothing)
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

@testset "DenseSurgeInteractionModel construction (default hidden_dims)" begin
    nwind = 3; nstations = 2; nlags = 4
    m = DenseSurgeInteractionModel(
        Dict{String, Any}("nlocations_output" => nstations, "nlocations_input" => nwind, "nlags" => nlags))

    @test m isa AbstractSurgeInteractionModel
    @test m isa AbstractSurgeModel
    @test m isa AbstractFluxModel

    fx = get_flux_model(m)
    @test fx isa DenseSurgeInteractionFlux
    chain = fx.chain
    @test chain isa Chain

    n_in = 3*nwind*nlags + nlags*nstations   # flattened forcing + flattened tide
    @test length(chain.layers) == 3          # Dense(in=>64), Dense(64=>32), Dense(32=>nstations)
    @test size(chain[1].weight) == (64, n_in)
    @test size(chain[2].weight) == (32, 64)
    @test size(chain[3].weight) == (nstations, 32)
    @test chain[1].σ === relu
    @test chain[3].σ === identity
end

@testset "DenseSurgeInteractionModel construction (custom hidden_dims / activation)" begin
    m = DenseSurgeInteractionModel(_dsi_settings(model_pars=Dict{String,Any}(
        "hidden_dims" => [8], "activation" => "tanh")))
    chain = get_flux_model(m).chain
    @test length(chain.layers) == 2   # Dense(in=>8), Dense(8=>nstations)
    @test chain[1].σ === tanh
end

@testset "DenseSurgeInteractionModel construction unknown activation errors" begin
    @test_throws ErrorException DenseSurgeInteractionModel(
        _dsi_settings(model_pars=Dict{String,Any}("activation" => "nope")))
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — two-tensor tuple (f_flat, t_lags), inherited from
# AbstractSurgeInteractionModel
# ──────────────────────────────────────────────────────────────────────────────

@testset "DenseSurgeInteractionModel preprocess" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m = DenseSurgeInteractionModel(_dsi_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    nvalid = ntimes - nlags + 1

    (f_flat, t_lags), output = preprocess(m, _dsi_input(nwind=nwind, nstations=nstations, ntimes=ntimes))

    @test size(f_flat) == (3*nwind*nlags, nvalid)
    @test size(t_lags) == (nlags, nstations, nvalid)
    @test eltype(f_flat) == Float32
    @test eltype(t_lags) == Float32
    @test haskey(output, "surge")
    @test size(output["surge"].values) == (nstations, nvalid)
    @test all(output["surge"].values .== 0f0)
end

@testset "DenseSurgeInteractionModel preprocess errors on missing tide" begin
    m = DenseSurgeInteractionModel(_dsi_settings())
    input = _dsi_input()
    delete!(input, "tide")
    @test_throws ErrorException preprocess(m, input)
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "DenseSurgeInteractionModel forward" begin
    nwind = 3; nstations = 2; nlags = 4; ntimes = 20
    m = DenseSurgeInteractionModel(_dsi_settings(nstations=nstations, nwind=nwind, nlags=nlags))

    f_flat = randn(Float32, 3*nwind*nlags, ntimes)
    t_lags = randn(Float32, nlags, nstations, ntimes)

    y = forward(m, (f_flat, t_lags))
    @test size(y) == (nstations, ntimes)
    @test eltype(y) == Float32

    # sanity check the concat-then-MLP behaviour directly against the chain
    t_flat = reshape(t_lags, nlags*nstations, ntimes)
    @test y ≈ get_flux_model(m).chain(vcat(f_flat, t_flat))
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model!
# ──────────────────────────────────────────────────────────────────────────────

@testset "DenseSurgeInteractionModel train_model!" begin
    nwind = 3; nstations = 2; ntimes = 50; nlags = 4
    settings = Dict{String, Any}(
        "nlocations_output" => nstations, "nlocations_input" => nwind, "nlags" => nlags)
    m      = DenseSurgeInteractionModel(settings)
    input  = _dsi_input(nwind=nwind, nstations=nstations, ntimes=ntimes)
    target = _dsi_target(nstations=nstations, ntimes=ntimes)

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

@testset "DenseSurgeInteractionModel predict" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m     = DenseSurgeInteractionModel(_dsi_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    input = _dsi_input(nwind=nwind, nstations=nstations, ntimes=ntimes)

    output = predict(m, input)
    nvalid = ntimes - nlags + 1

    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "surge")
    @test size(output["surge"].values) == (nstations, nvalid)
    @test !all(output["surge"].values .== 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip
# ──────────────────────────────────────────────────────────────────────────────

@testset "DenseSurgeInteractionModel save/load params" begin
    m  = DenseSurgeInteractionModel(_dsi_settings())
    chain = get_flux_model(m).chain

    W_orig = [copy(l.weight) for l in chain.layers]

    fn = joinpath(temp_dir, "dense_surge_interaction_params.jld2")
    save_params(m, fn)
    @test isfile(fn)
    @test_throws ErrorException save_params(m, fn)
    save_params(m, fn; overwrite=true)

    for l in chain.layers
        l.weight .= 0f0
    end

    load_params!(m, fn)
    for (l, W) in zip(chain.layers, W_orig)
        @test l.weight ≈ W
    end
end
