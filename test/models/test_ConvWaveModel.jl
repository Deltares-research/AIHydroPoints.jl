using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# nlags=4 → nchannel must have length 2 (2^2=4)
# ──────────────────────────────────────────────────────────────────────────────

const WAVE_NWIND      = 4
const WAVE_NSTATIONS  = 3
const WAVE_NTIMES     = 80
const WAVE_NLAGS      = 4   # 2^2 = 4 ✓

function wave_make_ts(values::Matrix{Float32}, quantity::String, names, lons, lats)
    ntimes = size(values, 2)
    times  = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

function wave_make_input(; nwind=WAVE_NWIND, nstations=WAVE_NSTATIONS, ntimes=WAVE_NTIMES)
    times    = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    w_names  = ["w$i" for i in 1:nwind]
    s_names  = ["s$i" for i in 1:nstations]
    w_lons   = Float64.(1:nwind);      w_lats  = fill(51.0, nwind)
    s_lons   = Float64.(4:4+nstations-1); s_lats = fill(52.0, nstations)

    u10  = TimeSeries(abs.(randn(Float32, nwind, ntimes)) .+ 5f0,      times, w_names, w_lons, w_lats, "wind_speed",     "test")
    udir = TimeSeries(rand(Float32, nwind, ntimes) .* 360f0,            times, w_names, w_lons, w_lats, "wind_direction", "test")
    swh  = TimeSeries(abs.(randn(Float32, nstations, ntimes)) .* 0.5f0, times, s_names, s_lons, s_lats, "wave_height",   "test")

    return Dict{String, TimeSeries}("wind_speed" => u10, "wind_direction" => udir, "wave_height" => swh)
end

function wave_make_settings(; nwind=WAVE_NWIND, nstations=WAVE_NSTATIONS, nlags=WAVE_NLAGS)
    s_names = ["s$i" for i in 1:nstations]
    return Dict{String, Any}(
        "nstations"        => nstations,
        "nwind"            => nwind,
        "nlags"            => nlags,
        "wind_scale"       => 0.5,
        "wave_scale"       => 3.0,
        "n_input_channels" => 8,
        "model_pars"       => Dict{String, Any}("nchannel" => [8, 1], "activation" => "relu"),
        "out_names"        => s_names,
        "out_lons"         => Float64.(4:4+nstations-1),
        "out_lats"         => fill(52.0, nstations),
        "out_quantity"     => "wave_height",
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvWaveModel construction" begin
    settings = wave_make_settings()
    m        = ConvWaveModel(settings)

    @test m isa AbstractWaveModel
    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_settings(m) === settings
    @test get_flux_model(m) isa Flux.Chain

    # nlags mismatch should throw
    bad = Dict{String,Any}("nstations"=>3,"nwind"=>4,"nlags"=>3,
                            "model_pars"=>Dict("nchannel"=>[8,1]))
    @test_throws AssertionError ConvWaveModel(bad)
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvWaveModel preprocess" begin
    m     = ConvWaveModel(wave_make_settings())
    input = wave_make_input()

    (x_station, x_input), output = preprocess(m, input)

    ntimes_valid = WAVE_NTIMES - WAVE_NLAGS + 1
    nsamples     = WAVE_NSTATIONS * ntimes_valid

    @test size(x_station) == (WAVE_NSTATIONS, nsamples)
    @test size(x_input)   == (WAVE_NLAGS, 2 * WAVE_NWIND, nsamples)
    @test eltype(x_input) == Float32

    @test haskey(output, "wave_height")
    @test size(output["wave_height"].values) == (WAVE_NSTATIONS, ntimes_valid)
    @test all(output["wave_height"].values .== 0f0)

    # Each column of x_station is one-hot
    @test all(sum(x_station; dims=1) .== 1)
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvWaveModel forward" begin
    nstations = WAVE_NSTATIONS; ntimes = 20
    ntimes_valid = ntimes - WAVE_NLAGS + 1
    nsamples     = nstations * ntimes_valid
    m = ConvWaveModel(wave_make_settings(nstations=nstations))

    x_station = Flux.onehotbatch(repeat(1:nstations, ntimes_valid), 1:nstations)
    x_input   = randn(Float32, WAVE_NLAGS, 2 * WAVE_NWIND, nsamples)
    y = forward(m, (x_station, x_input))

    @test size(y) == (nstations, 1, ntimes_valid)
    @test eltype(y) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model!
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvWaveModel train_model!" begin
    input  = wave_make_input()
    target = wave_make_input()   # target uses same wave_height values

    # Without out_names pre-set (train_model! should populate)
    settings = Dict{String, Any}(
        "nstations"        => WAVE_NSTATIONS,
        "nwind"            => WAVE_NWIND,
        "nlags"            => WAVE_NLAGS,
        "n_input_channels" => 8,
        "model_pars"       => Dict{String, Any}("nchannel" => [8, 1], "activation" => "relu"),
    )
    m  = ConvWaveModel(settings)
    ts = TrainingSettings(nepochs=3, nbatches=32, learning_rate=1e-3)

    train_losses, val_losses = train_model!(m, ts, input, target)

    @test haskey(m.settings, "out_names")
    @test haskey(m.settings, "nstations")
    @test haskey(m.settings, "nwind")
    @test length(train_losses) == ts.nepochs
    @test eltype(train_losses) == Float32
    @test all(train_losses .>= 0f0)
    @test isempty(val_losses)

    # With validation split
    ts_val = TrainingSettings(nepochs=3, nbatches=32, learning_rate=1e-3, validation_split=0.2)
    m2 = ConvWaveModel(Dict{String, Any}(
        "nstations" => WAVE_NSTATIONS, "nwind" => WAVE_NWIND, "nlags" => WAVE_NLAGS,
        "n_input_channels" => 8,
        "model_pars" => Dict{String, Any}("nchannel" => [8, 1], "activation" => "relu"),
    ))
    _, val_losses2 = train_model!(m2, ts_val, input, target)
    @test length(val_losses2) == ts_val.nepochs
    @test all(val_losses2 .>= 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvWaveModel predict" begin
    m     = ConvWaveModel(wave_make_settings())
    input = wave_make_input()

    output = predict(m, input)

    ntimes_valid = WAVE_NTIMES - WAVE_NLAGS + 1
    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "wave_height")
    @test size(output["wave_height"].values) == (WAVE_NSTATIONS, ntimes_valid)
    @test eltype(output["wave_height"].values) == Float32
    @test !all(output["wave_height"].values .== 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvWaveModel save/load params" begin
    m  = ConvWaveModel(wave_make_settings())
    fn = joinpath(temp_dir, "conv_wave_params.jld2")

    save_params(m, fn)
    @test isfile(fn)
    @test_throws ErrorException save_params(m, fn)
    save_params(m, fn; overwrite=true)

    bad_path = joinpath(temp_dir, "nonexistent_dir", "params.jld2")
    @test_throws ErrorException save_params(m, bad_path)

    m2 = ConvWaveModel(wave_make_settings())
    load_params!(m2, fn)
    @test Flux.state(get_flux_model(m2)) == Flux.state(get_flux_model(m))
end
