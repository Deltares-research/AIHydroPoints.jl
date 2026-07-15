using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — nlags=4, nchannel length=2 (2^2=4)
# ──────────────────────────────────────────────────────────────────────────────

const DON_NWIND     = 4
const DON_NSTATIONS = 3
const DON_NTIMES    = 80
const DON_NLAGS     = 4   # 2^2 = 4 ✓

function don_wave_make_input(; nwind=DON_NWIND, nstations=DON_NSTATIONS, ntimes=DON_NTIMES)
    times   = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    w_names = ["w$i" for i in 1:nwind]
    s_names = ["s$i" for i in 1:nstations]
    w_lons  = Float64.(1:nwind);         w_lats = fill(51.0, nwind)
    s_lons  = Float64.(4:4+nstations-1); s_lats = fill(52.0, nstations)

    u10  = TimeSeries(abs.(randn(Float32, nwind, ntimes)) .+ 5f0,      times, w_names, w_lons, w_lats, "wind_speed",     "test")
    udir = TimeSeries(rand(Float32, nwind, ntimes) .* 360f0,            times, w_names, w_lons, w_lats, "wind_direction", "test")
    swh  = TimeSeries(abs.(randn(Float32, nstations, ntimes)) .* 0.5f0, times, s_names, s_lons, s_lats, "wave_height",   "test")

    return Dict{String, TimeSeries}("wind_speed" => u10, "wind_direction" => udir, "wave_height" => swh)
end

function don_wave_make_settings(; nwind=DON_NWIND, nstations=DON_NSTATIONS, nlags=DON_NLAGS)
    s_names = ["s$i" for i in 1:nstations]
    return Dict{String, Any}(
        "nlocations_output"    => nstations,
        "nlocations_input"        => nwind,
        "nlags"        => nlags,
        "wind_scale"   => 0.5,
        "wave_scale"   => 3.0,
        "model_pars"   => Dict{String, Any}("nchannel" => [8, 1], "activation" => "relu"),
        "out_names"    => s_names,
        "out_lons"     => Float64.(4:4+nstations-1),
        "out_lats"     => fill(52.0, nstations),
        "out_quantity" => "wave_height",
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────

@testset "DeepONetWaveModel construction" begin
    settings = don_wave_make_settings()
    m        = DeepONetWaveModel(settings)

    @test m isa AbstractWaveModel
    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_settings(m) === settings
    @test get_flux_model(m) isa DeepONetWaveFlux

    bad = Dict{String,Any}("nlocations_output"=>3,"nlocations_input"=>4,"nlags"=>3,
                            "model_pars"=>Dict("nchannel"=>[8,1]))
    @test_throws AssertionError DeepONetWaveModel(bad)
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — inherited from AbstractWaveModel
# ──────────────────────────────────────────────────────────────────────────────

@testset "DeepONetWaveModel preprocess" begin
    m     = DeepONetWaveModel(don_wave_make_settings())
    input = don_wave_make_input()

    (x_station, x_input), output = preprocess(m, input)

    ntimes_valid = DON_NTIMES - DON_NLAGS + 1
    nsamples     = DON_NSTATIONS * ntimes_valid

    @test size(x_station) == (DON_NSTATIONS, nsamples)
    @test size(x_input)   == (DON_NLAGS, 2 * DON_NWIND, nsamples)
    @test eltype(x_input) == Float32
    @test haskey(output, "wave_height")
    @test size(output["wave_height"].values) == (DON_NSTATIONS, ntimes_valid)
    @test all(output["wave_height"].values .== 0f0)
    @test all(sum(x_station; dims=1) .== 1)
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "DeepONetWaveModel forward" begin
    nstations = DON_NSTATIONS; ntimes = 20
    ntimes_valid = ntimes - DON_NLAGS + 1
    nsamples     = nstations * ntimes_valid
    m = DeepONetWaveModel(don_wave_make_settings(nstations=nstations))

    x_station = Flux.onehotbatch(repeat(1:nstations, ntimes_valid), 1:nstations)
    x_input   = randn(Float32, DON_NLAGS, 2 * DON_NWIND, nsamples)
    y = forward(m, (x_station, x_input))

    @test size(y) == (1, nsamples)   # (1, nstations*ntimes_valid) — reshaped in postprocess!
    @test eltype(y) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model!
# ──────────────────────────────────────────────────────────────────────────────

@testset "DeepONetWaveModel train_model!" begin
    input  = don_wave_make_input()
    target = don_wave_make_input()

    settings = Dict{String, Any}(
        "nlocations_output"  => DON_NSTATIONS,
        "nlocations_input"      => DON_NWIND,
        "nlags"      => DON_NLAGS,
        "model_pars" => Dict{String, Any}("nchannel" => [8, 1], "activation" => "relu"),
    )
    m  = DeepONetWaveModel(settings)
    ts = TrainingSettings(nepochs=3, nbatches=32, learning_rate=1e-3)

    train_losses, val_losses = train_model!(m, ts, input, target)

    @test haskey(m.settings, "out_names")
    @test haskey(m.settings, "nlocations_output")
    @test haskey(m.settings, "nlocations_input")
    @test length(train_losses) == ts.nepochs
    @test eltype(train_losses) == Float32
    @test all(train_losses .>= 0f0)
    @test isempty(val_losses)

    ts_val = TrainingSettings(nepochs=3, nbatches=32, learning_rate=1e-3, validation_split=0.2)
    m2 = DeepONetWaveModel(Dict{String, Any}(
        "nlocations_output"  => DON_NSTATIONS,
        "nlocations_input"      => DON_NWIND,
        "nlags"      => DON_NLAGS,
        "model_pars" => Dict{String, Any}("nchannel" => [8, 1], "activation" => "relu"),
    ))
    _, val_losses2 = train_model!(m2, ts_val, input, target)
    @test length(val_losses2) == ts_val.nepochs
    @test all(val_losses2 .>= 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

@testset "DeepONetWaveModel predict" begin
    m     = DeepONetWaveModel(don_wave_make_settings())
    input = don_wave_make_input()

    output = predict(m, input)

    ntimes_valid = DON_NTIMES - DON_NLAGS + 1
    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "wave_height")
    @test size(output["wave_height"].values) == (DON_NSTATIONS, ntimes_valid)
    @test eltype(output["wave_height"].values) == Float32
    @test all(isfinite, output["wave_height"].values)   # relu+no-bias can zero out; just check finite
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip
# ──────────────────────────────────────────────────────────────────────────────

@testset "DeepONetWaveModel save/load params" begin
    m  = DeepONetWaveModel(don_wave_make_settings())
    fn = joinpath(temp_dir, "don_wave_params.jld2")

    save_params(m, fn)
    @test isfile(fn)
    @test_throws ErrorException save_params(m, fn)
    save_params(m, fn; overwrite=true)

    bad_path = joinpath(temp_dir, "nonexistent_dir", "params.jld2")
    @test_throws ErrorException save_params(m, bad_path)

    m2 = DeepONetWaveModel(don_wave_make_settings())
    load_params!(m2, fn)
    @test Flux.state(get_flux_model(m2)) == Flux.state(get_flux_model(m))
end
