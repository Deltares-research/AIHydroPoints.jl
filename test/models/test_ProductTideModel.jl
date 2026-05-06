using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (shared with test_DeepONetTideModel.jl naming convention)
# ──────────────────────────────────────────────────────────────────────────────

function prod_make_ts(values::Matrix{Float32}, quantity::String;
                      lons=nothing, lats=nothing)
    nstations, ntimes = size(values)
    times = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names = ["p$i" for i in 1:nstations]
    lons  = isnothing(lons) ? Float64.(3.0 .+ (1:nstations)) : lons
    lats  = isnothing(lats) ? Float64.(51.0 .+ (1:nstations)) : lats
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

function prod_make_input(; nstations=3, ntimes=50,
                          lons=[3.5, 4.1, 5.2], lats=[51.4, 51.9, 52.3])
    ts = prod_make_ts(randn(Float32, nstations, ntimes), "waterlevel"; lons, lats)
    return Dict{String, TimeSeries}("waterlevel" => ts)
end

function prod_make_settings(; nstations=3, freqs=["M2","S2","K1"],
                              lons=[3.5, 4.1, 5.2], lats=[51.4, 51.9, 52.3])
    return Dict{String, Any}(
        "freqs"        => freqs,
        "model_pars"   => Dict{String, Any}("nfeats" => 8, "nlayers" => 2),
        "out_names"    => ["p$i" for i in 1:nstations],
        "out_lons"     => lons,
        "out_lats"     => lats,
        "out_quantity" => "waterlevel",
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────

@testset "ProductTideModel construction" begin
    settings = prod_make_settings()
    m        = ProductTideModel(settings)

    @test m isa AbstractTideModel
    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_settings(m) === settings
    @test get_flux_model(m) isa ProductTideFlux

    # Default model_pars (nfeats=64, nlayers=3)
    s2 = Dict{String, Any}("freqs" => ["M2","S2"])
    m2 = ProductTideModel(s2)
    @test m2 isa ProductTideModel
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — inherited from AbstractTideModel
# ──────────────────────────────────────────────────────────────────────────────

@testset "ProductTideModel preprocess" begin
    nstations = 3; ntimes = 50; freqs = ["M2","S2","K1"]; nfreqs = length(freqs)
    m     = ProductTideModel(prod_make_settings(nstations=nstations, freqs=freqs))
    input = prod_make_input(nstations=nstations, ntimes=ntimes)

    (x_station, x_doodson), output = preprocess(m, input)

    @test size(x_station) == (4, nstations, ntimes)
    @test size(x_doodson) == (2 * nfreqs, ntimes)
    @test eltype(x_station) == Float32
    @test eltype(x_doodson) == Float32
    @test haskey(output, "waterlevel")
    @test size(output["waterlevel"].values) == (nstations, ntimes)
    @test all(output["waterlevel"].values .== 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "ProductTideModel forward" begin
    nstations = 3; ntimes = 20; freqs = ["M2","S2","K1"]; nfreqs = length(freqs)
    m = ProductTideModel(prod_make_settings(nstations=nstations, freqs=freqs))

    x_station = randn(Float32, 4, nstations, ntimes)
    x_doodson = randn(Float32, 2 * nfreqs, ntimes)
    y = forward(m, (x_station, x_doodson))

    @test size(y) == (nstations, 1, ntimes)
    @test eltype(y) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — inherited from AbstractTideModel
# ──────────────────────────────────────────────────────────────────────────────

@testset "ProductTideModel train_model!" begin
    nstations = 3; ntimes = 60; freqs = ["M2","S2","K1"]
    settings = Dict{String, Any}(
        "freqs"      => freqs,
        "model_pars" => Dict{String, Any}("nfeats" => 8, "nlayers" => 2),
    )
    m      = ProductTideModel(settings)
    input  = prod_make_input(nstations=nstations, ntimes=ntimes)
    target = prod_make_input(nstations=nstations, ntimes=ntimes)

    ts = TrainingSettings(nepochs=3, nbatches=16, learning_rate=1e-3)
    train_losses, val_losses = train_model!(m, ts, input, target)

    @test haskey(m.settings, "out_names")
    @test haskey(m.settings, "out_lons")
    @test haskey(m.settings, "out_lats")
    @test haskey(m.settings, "out_quantity")
    @test length(train_losses) == ts.nepochs
    @test eltype(train_losses) == Float32
    @test all(train_losses .>= 0f0)
    @test isempty(val_losses)

    # With validation split
    ts_val = TrainingSettings(nepochs=3, nbatches=16, learning_rate=1e-3, validation_split=0.2)
    m2 = ProductTideModel(Dict{String, Any}(
        "freqs" => freqs,
        "model_pars" => Dict{String, Any}("nfeats" => 8, "nlayers" => 2),
    ))
    _, val_losses2 = train_model!(m2, ts_val, input, target)
    @test length(val_losses2) == ts_val.nepochs
    @test all(val_losses2 .>= 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

@testset "ProductTideModel predict" begin
    nstations = 3; ntimes = 50; freqs = ["M2","S2","K1"]
    m     = ProductTideModel(prod_make_settings(nstations=nstations, freqs=freqs))
    input = prod_make_input(nstations=nstations, ntimes=ntimes)

    output = predict(m, input)

    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "waterlevel")
    @test size(output["waterlevel"].values) == (nstations, ntimes)
    @test eltype(output["waterlevel"].values) == Float32
    @test !all(output["waterlevel"].values .== 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip
# ──────────────────────────────────────────────────────────────────────────────

@testset "ProductTideModel save/load params" begin
    m  = ProductTideModel(prod_make_settings())
    fn = joinpath(temp_dir, "product_tide_params.jld2")

    save_params(m, fn)
    @test isfile(fn)
    @test_throws ErrorException save_params(m, fn)
    save_params(m, fn; overwrite=true)

    bad_path = joinpath(temp_dir, "nonexistent_dir", "params.jld2")
    @test_throws ErrorException save_params(m, bad_path)

    m2 = ProductTideModel(prod_make_settings())
    load_params!(m2, fn)
    @test Flux.state(get_flux_model(m2)) == Flux.state(get_flux_model(m))
end
