using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (same as test_LinearSurgeModel.jl)
# ──────────────────────────────────────────────────────────────────────────────

function conv_make_ts(values::Matrix{Float32}, quantity::String)
    nstations, ntimes = size(values)
    times = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names = ["s$i" for i in 1:nstations]
    lons  = Float64.(1:nstations)
    lats  = Float64.(51 .+ (1:nstations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

function conv_make_surge_input(; nwind=3, ntimes=30)
    sx = conv_make_ts(randn(Float32, nwind, ntimes), "stress_x")
    sy = conv_make_ts(randn(Float32, nwind, ntimes), "stress_y")
    p  = conv_make_ts(ones(Float32, nwind, ntimes) .* 1.0f5, "pressure")
    return Dict{String, TimeSeries}("stress_x" => sx, "stress_y" => sy, "pressure" => p)
end

function conv_make_surge_target(; nstations=2, ntimes=30)
    surge = conv_make_ts(randn(Float32, nstations, ntimes), "surge")
    return Dict{String, TimeSeries}("surge" => surge)
end

function conv_make_settings(; nstations=2, nwind=3, nlags=4)
    return Dict{String, Any}(
        "nstations"    => nstations,
        "nwind"        => nwind,
        "nlags"        => nlags,
        "out_names"    => ["s$i" for i in 1:nstations],
        "out_lons"     => Float64.(1:nstations),
        "out_lats"     => Float64.(51 .+ (1:nstations)),
        "out_quantity" => "surge",
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvSurgeModel construction" begin
    settings = Dict{String, Any}("nstations" => 2, "nwind" => 3, "nlags" => 4)
    m = ConvSurgeModel(settings)

    @test m isa AbstractSurgeModel
    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_settings(m) === settings

    # Default channels [32, 16]: chain has reshape + 2 Conv + flatten + Dense
    chain = get_flux_model(m)
    @test chain isa Chain

    # With explicit model_pars
    s2 = Dict{String, Any}(
        "nstations"  => 2, "nwind" => 3, "nlags" => 4,
        "model_pars" => Dict{String, Any}("channels" => [8], "filtersize" => 5),
    )
    m2 = ConvSurgeModel(s2)
    @test m2 isa ConvSurgeModel
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — inherited from AbstractSurgeModel
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvSurgeModel preprocess" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m     = ConvSurgeModel(conv_make_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    nvalid = ntimes - nlags + 1

    input = conv_make_surge_input(nwind=nwind, ntimes=ntimes)
    tensor, output = preprocess(m, input)

    @test size(tensor) == (1, 3*nwind, nlags, nvalid)
    @test eltype(tensor) == Float32
    @test haskey(output, "surge")
    @test size(output["surge"].values) == (nstations, nvalid)
    @test all(output["surge"].values .== 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvSurgeModel forward" begin
    nwind = 3; nstations = 2; nlags = 4; ntimes = 20
    m = ConvSurgeModel(conv_make_settings(nstations=nstations, nwind=nwind, nlags=nlags))

    x = randn(Float32, 1, 3*nwind, nlags, ntimes)
    y = forward(m, x)

    @test size(y) == (nstations, 1, ntimes)
    @test eltype(y) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — inherited from AbstractSurgeModel
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvSurgeModel train_model!" begin
    nwind = 3; nstations = 2; ntimes = 50; nlags = 4
    settings = Dict{String, Any}("nstations" => nstations, "nwind" => nwind, "nlags" => nlags)
    m        = ConvSurgeModel(settings)
    input    = conv_make_surge_input(nwind=nwind, ntimes=ntimes)
    target   = conv_make_surge_target(nstations=nstations, ntimes=ntimes)

    ts = TrainingSettings(nepochs=3, nbatches=16, learning_rate=1e-3)
    train_losses, val_losses = train_model!(m, ts, input, target)

    @test haskey(m.settings, "out_names")
    @test length(train_losses) == ts.nepochs
    @test eltype(train_losses) == Float32
    @test all(train_losses .>= 0f0)
    @test isempty(val_losses)

    # With validation split
    ts_val = TrainingSettings(nepochs=3, nbatches=16, learning_rate=1e-3, validation_split=0.2)
    m2 = ConvSurgeModel(Dict{String, Any}("nstations" => nstations, "nwind" => nwind, "nlags" => nlags))
    _, val_losses2 = train_model!(m2, ts_val, input, target)
    @test length(val_losses2) == ts_val.nepochs
    @test all(val_losses2 .>= 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

@testset "ConvSurgeModel predict" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m     = ConvSurgeModel(conv_make_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    input = conv_make_surge_input(nwind=nwind, ntimes=ntimes)

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

@testset "ConvSurgeModel save/load params" begin
    m  = ConvSurgeModel(conv_make_settings())

    fn = joinpath(temp_dir, "conv_surge_params.jld2")
    save_params(m, fn)
    @test isfile(fn)

    @test_throws ErrorException save_params(m, fn)
    save_params(m, fn; overwrite=true)

    # Zero out weights, reload, check restoration
    chain = get_flux_model(m)
    for layer in chain.layers
        if layer isa Conv
            layer.weight .= 0f0
        elseif layer isa Dense
            layer.weight .= 0f0
            layer.bias   .= 0f0
        end
    end

    load_params!(m, fn)

    # After reload, forward pass should give non-zero output (original weights restored)
    x = randn(Float32, 1, 3*3, 4, 5)   # 1, 3*nwind, nlags, ntimes
    @test !all(forward(m, x) .== 0f0)
end
