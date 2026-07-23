using Test
using AIHydroPoints
using Dates
using Flux
using Statistics: mean, std

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic data
# ──────────────────────────────────────────────────────────────────────────────

function cim_make_ts(values::Matrix{Float32}, quantity::String; nstations=size(values,1))
    ntimes = size(values, 2)
    times  = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names  = ["s$i" for i in 1:nstations]
    lons   = Float64.(3.0 .+ (1:nstations))
    lats   = Float64.(51.0 .+ (1:nstations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

"""Build a minimal input + target dict for unit tests."""
function cim_make_data(; nstations=3, ntimes=60)
    tide_vals  = randn(Float32, nstations, ntimes) .* 0.5f0
    surge_vals = randn(Float32, nstations, ntimes) .* 0.1f0
    wl_vals    = tide_vals .+ surge_vals .+ randn(Float32, nstations, ntimes) .* 0.02f0
    input = Dict{String, TimeSeries}(
        "tide"  => cim_make_ts(tide_vals,  "tide";  nstations),
        "surge" => cim_make_ts(surge_vals, "surge"; nstations),
    )
    target = Dict{String, TimeSeries}(
        "waterlevel" => cim_make_ts(wl_vals, "waterlevel"; nstations),
    )
    return input, target
end

# nlags=8, channels=[32,16,1] → nlags == 2^3 == 8 ✓
function make_cim_settings(; nstations=3)
    return Dict{String, Any}(
        "nlocations_output"  => nstations,
        "nlags"      => 8,
        "model_pars" => Dict{String, Any}("channels" => [32, 16, 1]),
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvInteractionModel construction" begin
    m = ConvInteractionModel(make_cim_settings())
    @test m isa AbstractInteractionModel
    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_settings(m) isa Dict{String, Any}
    @test get_flux_model(m) isa ConvInteractionFlux

    # nlags / channels mismatch should error
    bad = Dict{String, Any}(
        "nlocations_output"  => 3,
        "nlags"      => 4,
        "model_pars" => Dict{String, Any}("channels" => [32, 16, 1]),  # 3 layers → 2^3=8 ≠ 4
    )
    @test_throws AssertionError ConvInteractionModel(bad)
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvInteractionModel preprocess" begin
    nstations = 3; ntimes = 50; nlags = 8
    m = ConvInteractionModel(make_cim_settings(nstations=nstations))
    input, _ = cim_make_data(nstations=nstations, ntimes=ntimes)

    (x_station, x_ts), output = preprocess(m, input)

    ntimes_valid = ntimes - nlags + 1
    nsamples     = nstations * ntimes_valid

    @test size(x_station) == (nstations, nsamples)
    @test size(x_ts)      == (nlags, 2, nsamples)
    @test eltype(x_ts) == Float32

    @test haskey(output, "waterlevel")
    @test size(output["waterlevel"].values) == (nstations, ntimes_valid)
    @test all(output["waterlevel"].values .== 0f0)
    @test length(get_times(output["waterlevel"])) == ntimes_valid
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvInteractionModel forward" begin
    nstations = 3; nlags = 8; ntimes = 14
    nsamples  = nstations * ntimes
    m = ConvInteractionModel(make_cim_settings(nstations=nstations))

    x_station = Float32.(Flux.onehotbatch(rand(1:nstations, nsamples), 1:nstations))
    x_ts      = randn(Float32, nlags, 2, nsamples)

    y = forward(m, (x_station, x_ts))

    @test size(y) == (1, nsamples)   # (1, nstations*ntimes) — reshaped in postprocess!
    @test eltype(y) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model!
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvInteractionModel train_model!" begin
    nstations = 3; ntimes = 80
    m = ConvInteractionModel(make_cim_settings(nstations=nstations))
    input, target = cim_make_data(nstations=nstations, ntimes=ntimes)

    ts = TrainingSettings(nepochs=3, nbatches=16, learning_rate=1e-3)
    train_losses, val_losses = train_model!(m, ts, input, target)

    @test length(train_losses) == ts.nepochs
    @test eltype(train_losses) == Float32
    @test all(train_losses .>= 0f0)
    @test isempty(val_losses)

    # Normalization stats populated
    s = get_settings(m)
    @test haskey(s, "input_mu")
    @test haskey(s, "input_std")
    @test haskey(s, "output_mu")
    @test haskey(s, "output_std")

    # Output metadata populated
    @test haskey(s, "out_names")
    @test haskey(s, "out_lons")
    @test haskey(s, "out_lats")
    @test haskey(s, "out_quantity")

    # With validation split
    ts_val = TrainingSettings(nepochs=3, nbatches=16, learning_rate=1e-3, validation_split=0.2)
    m2 = ConvInteractionModel(make_cim_settings(nstations=nstations))
    train_losses2, val_losses2 = train_model!(m2, ts_val, input, target)
    @test length(val_losses2) == ts_val.nepochs
    @test all(val_losses2 .>= 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# predict (end-to-end)
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvInteractionModel predict" begin
    nstations = 3; ntimes = 60; nlags = 8
    m = ConvInteractionModel(make_cim_settings(nstations=nstations))
    input, target = cim_make_data(nstations=nstations, ntimes=ntimes)

    ts = TrainingSettings(nepochs=2, nbatches=16, learning_rate=1e-3)
    train_model!(m, ts, input, target)

    output = predict(m, input)

    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "waterlevel")
    @test size(output["waterlevel"].values) == (nstations, ntimes - nlags + 1)
    @test eltype(output["waterlevel"].values) == Float32
    @test !all(output["waterlevel"].values .== 0f0)

    # Output times aligned to tide times[nlags:end]
    expected_times = get_times(input["tide"])[nlags:end]
    @test get_times(output["waterlevel"]) == expected_times
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvInteractionModel save/load params" begin
    m  = ConvInteractionModel(make_cim_settings())
    fn = joinpath(temp_dir, "conv_interaction_params.jld2")

    save_params(m, fn)
    @test isfile(fn)
    @test_throws ErrorException save_params(m, fn)
    save_params(m, fn; overwrite=true)

    bad_path = joinpath(temp_dir, "nonexistent_dir", "params.jld2")
    @test_throws ErrorException save_params(m, bad_path)

    m2 = ConvInteractionModel(make_cim_settings())
    load_params!(m2, fn)
    @test Flux.state(get_flux_model(m2)) == Flux.state(get_flux_model(m))
end
