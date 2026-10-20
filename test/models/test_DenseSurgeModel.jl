using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (uniquely named to avoid clashing with other model test files)
# ──────────────────────────────────────────────────────────────────────────────

function _ds_ts(values::Matrix{Float32}, quantity::String)
    nstations, ntimes = size(values)
    times = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names = ["s$i" for i in 1:nstations]
    lons  = Float64.(1:nstations)
    lats  = Float64.(51 .+ (1:nstations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

"""Forcing-only input dict for DenseSurgeModel (stress keys by default)."""
function _ds_input(; nwind=3, ntimes=30, use_wind_keys=false)
    sx = _ds_ts(randn(Float32, nwind, ntimes), "stress_x")
    sy = _ds_ts(randn(Float32, nwind, ntimes), "stress_y")
    pressure = _ds_ts(ones(Float32, nwind, ntimes) .* 1.0f5, "pressure")
    keys = use_wind_keys ? ("wind_x", "wind_y") : ("stress_x", "stress_y")
    return Dict{String, TimeSeries}(
        keys[1]    => sx,
        keys[2]    => sy,
        "pressure" => pressure,
    )
end

_ds_target(; nstations=2, ntimes=30) =
    Dict{String, TimeSeries}("surge" => _ds_ts(randn(Float32, nstations, ntimes), "surge"))

"""Settings with output metadata pre-filled (for tests that bypass train_model!)."""
function _ds_settings(; nstations=2, nwind=3, nlags=4, model_pars=nothing)
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

@testset "DenseSurgeModel construction (default hidden_dims)" begin
    settings = Dict{String, Any}("nlocations_output" => 2, "nlocations_input" => 3, "nlags" => 4)
    m = DenseSurgeModel(settings)

    @test m isa AbstractSurgeModel
    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_settings(m) === settings

    chain = get_flux_model(m)
    @test chain isa Chain
    # only, Dense(in=>64), Dense(64=>32), Dense(32=>nstations)  → 4 elements
    @test length(chain.layers) == 4
    @test chain[2] isa Dense && size(chain[2].weight) == (64, 3*3*4)
    @test chain[3] isa Dense && size(chain[3].weight) == (32, 64)
    @test chain[4] isa Dense && size(chain[4].weight) == (2, 32)
    @test chain[2].σ === relu
    @test chain[4].σ === identity
end

@testset "DenseSurgeModel construction (custom hidden_dims / activation)" begin
    m = DenseSurgeModel(_ds_settings(model_pars=Dict{String,Any}(
        "hidden_dims" => [8], "activation" => "tanh")))
    chain = get_flux_model(m)
    @test length(chain.layers) == 3   # only, Dense(in=>8), Dense(8=>nstations)
    @test chain[2].σ === tanh
end

@testset "DenseSurgeModel construction (empty hidden_dims ≡ LinearSurgeModel)" begin
    m = DenseSurgeModel(_ds_settings(model_pars=Dict{String,Any}("hidden_dims" => Int[])))
    chain = get_flux_model(m)
    @test length(chain.layers) == 2   # only, Dense(in=>nstations)
    @test size(chain[2].weight) == (2, 3*3*4)
end

@testset "DenseSurgeModel construction unknown activation errors" begin
    @test_throws ErrorException DenseSurgeModel(
        _ds_settings(model_pars=Dict{String,Any}("activation" => "nope")))
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — identical layout to LinearSurgeModel
# ──────────────────────────────────────────────────────────────────────────────

@testset "DenseSurgeModel preprocess" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m     = DenseSurgeModel(_ds_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    nvalid = ntimes - nlags + 1

    for use_wind_keys in (false, true)
        input = _ds_input(nwind=nwind, ntimes=ntimes, use_wind_keys=use_wind_keys)
        (x_flat,), output = preprocess(m, input)

        @test size(x_flat) == (3*nwind*nlags, nvalid)
        @test eltype(x_flat) == Float32
        @test haskey(output, "surge")
        @test size(output["surge"].values) == (nstations, nvalid)
        @test all(output["surge"].values .== 0f0)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "DenseSurgeModel forward" begin
    nwind = 3; nstations = 2; nlags = 4; ntimes = 20
    m = DenseSurgeModel(_ds_settings(nstations=nstations, nwind=nwind, nlags=nlags))

    x = zeros(Float32, 3*nwind*nlags, ntimes)
    y = forward(m, (x,))

    @test size(y) == (nstations, ntimes)
    @test eltype(y) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model!
# ──────────────────────────────────────────────────────────────────────────────

@testset "DenseSurgeModel train_model!" begin
    nwind = 3; nstations = 2; ntimes = 50; nlags = 4
    settings = Dict{String, Any}("nlocations_output" => nstations, "nlocations_input" => nwind, "nlags" => nlags)
    m        = DenseSurgeModel(settings)
    input    = _ds_input(nwind=nwind, ntimes=ntimes)
    target   = _ds_target(nstations=nstations, ntimes=ntimes)

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

@testset "DenseSurgeModel predict" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m      = DenseSurgeModel(_ds_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    input  = _ds_input(nwind=nwind, ntimes=ntimes)

    output = predict(m, input)
    nvalid = ntimes - nlags + 1

    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "surge")
    @test size(output["surge"].values) == (nstations, nvalid)
    @test eltype(output["surge"].values) == Float32
    @test !all(output["surge"].values .== 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip
# ──────────────────────────────────────────────────────────────────────────────

@testset "DenseSurgeModel save/load params" begin
    m  = DenseSurgeModel(_ds_settings())
    chain = get_flux_model(m)

    W_orig = [copy(l.weight) for l in chain.layers[2:end]]

    fn = joinpath(temp_dir, "dense_surge_params.jld2")
    save_params(m, fn)
    @test isfile(fn)
    @test_throws ErrorException save_params(m, fn)
    save_params(m, fn; overwrite=true)

    for l in chain.layers[2:end]
        l.weight .= 0f0
    end

    load_params!(m, fn)
    for (l, W) in zip(chain.layers[2:end], W_orig)
        @test l.weight ≈ W
    end
end
