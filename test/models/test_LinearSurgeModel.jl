using Test
using AIHydroPoints
using Dates
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""Build a minimal TimeSeries for testing (nstations × ntimes)."""
function make_ts(values::Matrix{Float32}, quantity::String)
    nstations, ntimes = size(values)
    times = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names = ["s$i" for i in 1:nstations]
    lons  = Float64.(1:nstations)
    lats  = Float64.(51 .+ (1:nstations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

"""Forcing-only input dict for LinearSurgeModel (stress keys by default)."""
function make_surge_input(; nwind=3, ntimes=30, use_wind_keys=false)
    sx = make_ts(randn(Float32, nwind, ntimes), "stress_x")
    sy = make_ts(randn(Float32, nwind, ntimes), "stress_y")
    pressure = make_ts(ones(Float32, nwind, ntimes) .* 1.0f5, "pressure")
    keys = use_wind_keys ? ("wind_x", "wind_y") : ("stress_x", "stress_y")
    return Dict{String, TimeSeries}(
        keys[1]    => sx,
        keys[2]    => sy,
        "pressure" => pressure,
    )
end

"""Target dict for LinearSurgeModel."""
function make_surge_target(; nstations=2, ntimes=30)
    surge = make_ts(randn(Float32, nstations, ntimes), "surge")
    return Dict{String, TimeSeries}("surge" => surge)
end

"""Settings with output metadata pre-filled (for tests that bypass train_model!)."""
function make_lsm_settings(; nstations=2, nwind=3, nlags=4)
    return Dict{String, Any}(
        "nlocations_output"    => nstations,
        "nlocations_input"        => nwind,
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

@testset "LinearSurgeModel construction" begin
    settings = Dict{String, Any}("nlocations_output" => 2, "nlocations_input" => 3, "nlags" => 4)
    m = LinearSurgeModel(settings)

    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_settings(m) === settings
    @test get_flux_model(m) isa Dense

    # Dense layer has correct dimensions: 3*nwind*nlags → nstations
    @test size(get_flux_model(m).weight) == (2, 3*3*4)
end

# ──────────────────────────────────────────────────────────────────────────────
# preprocess — metadata comes from settings, not from input
# ──────────────────────────────────────────────────────────────────────────────

@testset "LinearSurgeModel preprocess" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m     = LinearSurgeModel(make_lsm_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    nvalid = ntimes - nlags + 1

    for use_wind_keys in (false, true)
        input = make_surge_input(nwind=nwind, ntimes=ntimes, use_wind_keys=use_wind_keys)
        tensor, output = preprocess(m, input)

        @test size(tensor) == (1, 3*nwind, nlags, nvalid)
        @test eltype(tensor) == Float32
        @test haskey(output, "surge")
        @test size(output["surge"].values) == (nstations, nvalid)
        @test all(output["surge"].values .== 0f0)
        @test length(get_times(output["surge"])) == nvalid
        @test get_names(output["surge"]) == m.settings["out_names"]
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# forward
# ──────────────────────────────────────────────────────────────────────────────

@testset "LinearSurgeModel forward" begin
    nwind = 3; nstations = 2; nlags = 4; ntimes = 20
    m = LinearSurgeModel(make_lsm_settings(nstations=nstations, nwind=nwind, nlags=nlags))

    x = zeros(Float32, 1, 3*nwind, nlags, ntimes)
    y = forward(m, x)

    @test size(y) == (nstations, 1, ntimes)
    @test eltype(y) == Float32
end

# ──────────────────────────────────────────────────────────────────────────────
# train_model! — metadata population + training loop
# ──────────────────────────────────────────────────────────────────────────────

@testset "LinearSurgeModel train_model!" begin
    nwind = 3; nstations = 2; ntimes = 50; nlags = 4
    # Start with no output metadata in settings
    settings = Dict{String, Any}("nlocations_output" => nstations, "nlocations_input" => nwind, "nlags" => nlags)
    m        = LinearSurgeModel(settings)
    input    = make_surge_input(nwind=nwind, ntimes=ntimes)
    target   = make_surge_target(nstations=nstations, ntimes=ntimes)

    ts = TrainingSettings(nepochs=3, nbatches=16, learning_rate=1e-3)
    train_losses, val_losses = train_model!(m, ts, input, target)

    # Metadata populated from target
    @test haskey(m.settings, "out_names")
    @test haskey(m.settings, "out_lons")
    @test haskey(m.settings, "out_lats")
    @test haskey(m.settings, "out_quantity")
    @test m.settings["out_names"] == get_names(target["surge"])

    # Returns one loss per epoch; no val split by default
    @test length(train_losses) == ts.nepochs
    @test eltype(train_losses) == Float32
    @test all(train_losses .>= 0f0)
    @test isempty(val_losses)

    # With validation_split
    ts_val = TrainingSettings(nepochs=3, nbatches=16, learning_rate=1e-3, validation_split=0.2)
    m2 = LinearSurgeModel(Dict{String,Any}("nlocations_output" => nstations, "nlocations_input" => nwind, "nlags" => nlags))
    train_losses2, val_losses2 = train_model!(m2, ts_val, input, target)
    @test length(val_losses2) == ts_val.nepochs
    @test eltype(val_losses2) == Float32
    @test all(val_losses2 .>= 0f0)
end

# ──────────────────────────────────────────────────────────────────────────────
# end-to-end predict (after train_model! sets metadata)
# ──────────────────────────────────────────────────────────────────────────────

@testset "LinearSurgeModel predict" begin
    nwind = 3; nstations = 2; ntimes = 30; nlags = 4
    m      = LinearSurgeModel(make_lsm_settings(nstations=nstations, nwind=nwind, nlags=nlags))
    input  = make_surge_input(nwind=nwind, ntimes=ntimes)

    output = predict(m, input)
    nvalid = ntimes - nlags + 1

    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "surge")
    @test size(output["surge"].values) == (nstations, nvalid)
    @test eltype(output["surge"].values) == Float32
    @test !all(output["surge"].values .== 0f0)   # Dense has random weights
end

# ──────────────────────────────────────────────────────────────────────────────
# save_params / load_params! round-trip
# ──────────────────────────────────────────────────────────────────────────────

@testset "LinearSurgeModel save/load params" begin
    m  = LinearSurgeModel(make_lsm_settings())

    W_orig = copy(get_flux_model(m).weight)
    b_orig = copy(get_flux_model(m).bias)

    fn = joinpath(temp_dir, "linear_surge_params.jld2")
    save_params(m, fn)
    @test isfile(fn)

    # error if file exists and overwrite=false (default)
    @test_throws ErrorException save_params(m, fn)

    # overwrite=true replaces the file
    save_params(m, fn; overwrite=true)
    @test isfile(fn)

    # error if parent directory does not exist
    bad_path = joinpath(temp_dir, "nonexistent_dir", "params.jld2")
    @test_throws ErrorException save_params(m, bad_path)

    get_flux_model(m).weight .= 0f0
    get_flux_model(m).bias   .= 0f0

    load_params!(m, fn)
    @test get_flux_model(m).weight ≈ W_orig
    @test get_flux_model(m).bias   ≈ b_orig
end

@testset "preprocess location alignment" begin
    settings = make_lsm_settings(; nstations=2, nwind=3, nlags=4)
    settings["in_names"] = ["w1", "w2", "w3"]
    settings["model_dir"] = temp_dir
    m = LinearSurgeModel(settings)

    ntimes = 20
    times  = collect(DateTime(2020,1,1) .+ Hour.(0:ntimes-1))
    make_wind_ts(names) = TimeSeries(
        randn(Float32, length(names), ntimes), times, names,
        Float64.(1:length(names)), Float64.(51 .+ (1:length(names))), "stress", "test")

    # correct order: works
    input_ok = Dict(
        "stress_x" => make_wind_ts(["w1","w2","w3"]),
        "stress_y" => make_wind_ts(["w1","w2","w3"]),
        "pressure" => make_wind_ts(["w1","w2","w3"]))
    @test_nowarn AIHydroPoints.preprocess(m, input_ok)

    # wrong order: reordered automatically, no error
    input_shuffled = Dict(
        "stress_x" => make_wind_ts(["w3","w1","w2"]),
        "stress_y" => make_wind_ts(["w3","w1","w2"]),
        "pressure" => make_wind_ts(["w3","w1","w2"]))
    @test_nowarn AIHydroPoints.preprocess(m, input_shuffled)

    # extra locations: silently dropped
    input_extra = Dict(
        "stress_x" => make_wind_ts(["w1","w2","w3","w4"]),
        "stress_y" => make_wind_ts(["w1","w2","w3","w4"]),
        "pressure" => make_wind_ts(["w1","w2","w3","w4"]))
    @test_nowarn AIHydroPoints.preprocess(m, input_extra)

    # missing location: readable error
    input_missing = Dict(
        "stress_x" => make_wind_ts(["w1","w2"]),
        "stress_y" => make_wind_ts(["w1","w2"]),
        "pressure" => make_wind_ts(["w1","w2"]))
    err = @test_throws ErrorException AIHydroPoints.preprocess(m, input_missing)
    @test occursin("missing", err.value.msg)
    @test occursin("w3", err.value.msg)
end
