using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

function attn_make_ts(values::Matrix{Float32}, quantity::String;
                 lons=nothing, lats=nothing)
    nstations, ntimes = size(values)
    times = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names = ["s$i" for i in 1:nstations]
    lons  = isnothing(lons) ? Float64.(1:nstations) : lons
    lats  = isnothing(lats) ? Float64.(51 .+ (1:nstations)) : lats
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

function attn_make_surge_input(; nwind=3, ntimes=50, use_wind_keys=false)
    sx = attn_make_ts(randn(Float32, nwind, ntimes), "stress_x")
    sy = attn_make_ts(randn(Float32, nwind, ntimes), "stress_y")
    pressure = attn_make_ts(ones(Float32, nwind, ntimes) .* 1.0f5, "pressure")
    keys = use_wind_keys ? ("wind_x", "wind_y") : ("stress_x", "stress_y")
    return Dict{String, TimeSeries}(
        keys[1]    => sx,
        keys[2]    => sy,
        "pressure" => pressure,
    )
end

function attn_make_surge_target(; nstations=2, ntimes=50,
                             lons=[3.5, 4.1], lats=[51.4, 51.9])
    surge = attn_make_ts(randn(Float32, nstations, ntimes), "surge"; lons, lats)
    return Dict{String, TimeSeries}("surge" => surge)
end

"""Minimal GraphNetwork connecting nwind input points to nstations output points."""
function attn_make_graph_network(; nwind=3, nstations=2,
                              in_lons=[3.0, 4.0, 5.0], in_lats=[51.0, 52.0, 53.0],
                              out_lons=[3.5, 4.1],     out_lats=[51.4, 51.9])
    in_points  = [(deg2rad(la), deg2rad(lo)) for (la, lo) in zip(in_lats,  in_lons)]
    out_points = [(deg2rad(la), deg2rad(lo)) for (la, lo) in zip(out_lats, out_lons)]
    adjacency  = ones(Float32, nwind, nstations)   # fully connected
    return GraphNetwork(in_points, out_points, adjacency)
end

"""Minimal model_pars for the attention architecture."""
function attn_make_model_pars(; nembed=8, nheads=2, nlayers_branch=1, nlayers_trunk=1,
                           nhidden_trunk=8, theta=1000.0)
    return Dict{String,Any}(
        "nembed"         => nembed,
        "theta"          => theta,
        "nheads"         => nheads,
        "nlayers_branch" => nlayers_branch,
        "nlayers_trunk"  => nlayers_trunk,
        "nhidden_trunk"  => nhidden_trunk,
    )
end

function attn_make_settings(; nstations=2, nwind=3, nlags=4,
                         out_lons=[3.5, 4.1], out_lats=[51.4, 51.9])
    return Dict{String, Any}(
        "nlocations_output"    => nstations,
        "nlocations_input"        => nwind,
        "nlags"        => nlags,
        "model_pars"   => attn_make_model_pars(),
        "out_names"    => ["s$i" for i in 1:nstations],
        "out_lons"     => out_lons,
        "out_lats"     => out_lats,
        "out_quantity" => "surge",
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────

@testset "AttentionSurgeModel construction" begin
    settings = attn_make_settings()
    gn       = attn_make_graph_network()
    m        = AttentionSurgeModel(settings, gn)

    @test m isa AbstractSurgeModel
    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_settings(m) === settings
    @test get_flux_model(m) isa AttentionSurgeFlux
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess
# ──────────────────────────────────────────────────────────────────────────────

@testset "AttentionSurgeModel preprocess" begin
    nwind = 3; nstations = 2; ntimes = 50; nlags = 4
    m     = AttentionSurgeModel(attn_make_settings(nstations=nstations, nwind=nwind, nlags=nlags),
                                attn_make_graph_network(nwind=nwind, nstations=nstations))
    nvalid = ntimes - nlags + 1

    for use_wind_keys in (false, true)
        input = attn_make_surge_input(nwind=nwind, ntimes=ntimes, use_wind_keys=use_wind_keys)
        (x_station, x_wind), output = preprocess(m, input)

        @test size(x_wind)    == (3*nwind, nlags, nvalid)
        @test size(x_station) == (6, nstations, nvalid)
        @test eltype(x_wind)    == Float32
        @test eltype(x_station) == Float32
        @test haskey(output, "surge")
        @test size(output["surge"].values) == (nstations, nvalid)
        @test all(output["surge"].values .== 0f0)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "AttentionSurgeModel forward" begin
    nwind = 3; nstations = 2; nlags = 4; ntimes = 20
    m = AttentionSurgeModel(attn_make_settings(nstations=nstations, nwind=nwind, nlags=nlags),
                            attn_make_graph_network(nwind=nwind, nstations=nstations))

    x_wind    = randn(Float32, 3*nwind, nlags, ntimes)
    x_station = randn(Float32, 6, nstations, ntimes)
    y = forward(m, (x_station, x_wind))

    @test size(y) == (nstations, ntimes)
    @test eltype(y) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model!
# ──────────────────────────────────────────────────────────────────────────────

@testset "AttentionSurgeModel train_model!" begin
    nwind = 3; nstations = 2; ntimes = 60; nlags = 4
    settings = Dict{String,Any}("nlocations_output" => nstations, "nlocations_input" => nwind,
                                "nlags" => nlags, "model_pars" => attn_make_model_pars())
    m      = AttentionSurgeModel(settings, attn_make_graph_network(nwind=nwind, nstations=nstations))
    input  = attn_make_surge_input(nwind=nwind, ntimes=ntimes)
    target = attn_make_surge_target(nstations=nstations, ntimes=ntimes)

    ts = TrainingSettings(nepochs=3, batch_size=16, learning_rate=1e-3)
    train_losses, val_losses = train_model!(m, ts, input, target)

    @test haskey(m.settings, "out_names")
    @test haskey(m.settings, "out_lons")
    @test haskey(m.settings, "out_lats")
    @test haskey(m.settings, "out_quantity")

    @test length(train_losses) == ts.nepochs
    @test eltype(train_losses) == Float32
    @test all(train_losses .>= 0f0)
    @test isempty(val_losses)

    # With validation_split
    ts_val = TrainingSettings(nepochs=3, batch_size=16, learning_rate=1e-3, validation_split=0.2)
    m2 = AttentionSurgeModel(
        Dict{String,Any}("nlocations_output" => nstations, "nlocations_input" => nwind,
                         "nlags" => nlags, "model_pars" => attn_make_model_pars()),
        attn_make_graph_network(nwind=nwind, nstations=nstations))
    train_losses2, val_losses2 = train_model!(m2, ts_val, input, target)
    @test length(val_losses2) == ts_val.nepochs
    @test all(val_losses2 .>= 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# predict (end-to-end)
# ──────────────────────────────────────────────────────────────────────────────

@testset "AttentionSurgeModel predict" begin
    nwind = 3; nstations = 2; ntimes = 50; nlags = 4
    m     = AttentionSurgeModel(attn_make_settings(nstations=nstations, nwind=nwind, nlags=nlags),
                                attn_make_graph_network(nwind=nwind, nstations=nstations))
    input = attn_make_surge_input(nwind=nwind, ntimes=ntimes)

    output = predict(m, input)
    nvalid = ntimes - nlags + 1

    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "surge")
    @test size(output["surge"].values) == (nstations, nvalid)
    @test eltype(output["surge"].values) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip
# ──────────────────────────────────────────────────────────────────────────────

@testset "AttentionSurgeModel save/load params" begin
    m  = AttentionSurgeModel(attn_make_settings(), attn_make_graph_network())
    fn = joinpath(temp_dir, "attention_surge_params.jld2")

    save_params(m, fn)
    @test isfile(fn)
    @test_throws ErrorException save_params(m, fn)
    save_params(m, fn; overwrite=true)

    bad_path = joinpath(temp_dir, "nonexistent_dir", "params.jld2")
    @test_throws ErrorException save_params(m, bad_path)

    # Load into a fresh model and verify weights match
    m2 = AttentionSurgeModel(attn_make_settings(), attn_make_graph_network())
    load_params!(m2, fn)
    @test Flux.state(get_flux_model(m2)) == Flux.state(get_flux_model(m))
end
